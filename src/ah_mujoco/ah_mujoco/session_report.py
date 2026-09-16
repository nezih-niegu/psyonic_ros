"""Turn a recorded session into a report.

    ros2 run ah_mujoco session_report --ros-args \\
        -p session:=/ws/vendor/session1.npz \\
        -p out:=/ws/vendor/session1.html

Or directly, without ROS:

    python3 -m ah_mujoco.session_report session1.npz session1.html

Produces one self-contained HTML file: no server, no internet, opens anywhere.

The report answers two questions separately, because a link can be good at one
and bad at the other:

  How faithfully did the robot reproduce the operator's motion?
  What did the operator pay in posture and effort to drive it?
"""

import sys

import numpy as np

from ah_mujoco.analysis import (
    COMFORT,
    exposure,
    jerk_exposure,
    joint_angles,
    rula_arm_score,
    static_loading,
    smoothness_ratio,
    tracking_error,
    transmission_bandwidth,
)

# Matches the UI's scale so a plot and the live display agree
OK = "#3fb950"
WARN = "#d29922"
BAD = "#f85149"
ACCENT = "#4cc2ff"
DIM = "#8b949e"


def _speed(x, dt):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return np.abs(np.gradient(x, dt))
    return np.linalg.norm(np.gradient(x, dt, axis=0), axis=1)


def _forward_kinematics(joints):
    """Arm joint angles -> tool position, using the same chain the IK uses."""
    try:
        from ah_mujoco.arm_teleop import (LITE6_LIMITS, LITE6_ORIGINS,
                                          UrdfChain)
    except ImportError:
        return np.asarray(joints, dtype=float)[:, :3]
    chain = UrdfChain(LITE6_ORIGINS, LITE6_LIMITS)
    out = np.zeros((len(joints), 3))
    for i, q in enumerate(joints):
        if not np.all(np.isfinite(q)):
            out[i] = out[i - 1] if i else 0.0
            continue
        out[i] = chain.fk(np.asarray(q, dtype=float))[:3, 3]
    return out


def _clean(x):
    """Forward-fill gaps so a dropout does not become a spike."""
    x = np.asarray(x, dtype=float).copy()
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False
    for c in range(x.shape[1]):
        col = x[:, c]
        good = np.where(np.isfinite(col))[0]
        if good.size == 0:
            continue
        col[: good[0]] = col[good[0]]
        for i in range(1, len(col)):
            if not np.isfinite(col[i]):
                col[i] = col[i - 1]
    return x[:, 0] if squeeze else x


def analyse(path):
    d = np.load(path)
    t = d["t"]
    dt = float(np.median(np.diff(t))) if t.size > 2 else 1 / 60.0

    limb = _clean(d["limb"])
    wrist = _clean(d["wrist"])
    arm = _clean(d["arm_joints"])
    clutch = d["clutch"]
    hand = _clean(d["hand_targets"])

    shoulder, elbow, wrist_b = limb[:, 0:3], limb[:, 3:6], limb[:, 6:9]
    ang = joint_angles(shoulder, elbow, wrist_b)

    # Operator input vs robot output, both as CARTESIAN speed. The robot's
    # joint angles are run through forward kinematics first: comparing a
    # wrist speed in m/s against the norm of six joint velocities in rad/s
    # compares different quantities and gives a meaninglessly low correlation
    # (0.15 on a link that was actually tracking well).
    #
    # Speed magnitude rather than position keeps this independent of the
    # hand-eye transform, which is a separate calibration and not necessarily
    # right.
    tcp = _forward_kinematics(arm)
    op_speed = _speed(wrist, dt)
    rob_speed = _speed(tcp, dt)

    engaged = clutch > 0.5
    fid = tracking_error(op_speed, rob_speed, dt)
    bw = transmission_bandwidth(op_speed, rob_speed, dt)
    smooth = smoothness_ratio(op_speed, rob_speed, dt)

    stress = {}
    for name, series in ang.items():
        lo, hi = COMFORT[name]
        stress[name] = exposure(series, lo, hi, dt)
    stress["static"] = static_loading(_speed(elbow, dt), dt)
    stress["jerk"] = jerk_exposure(ang["elbow_flexion"], dt)
    rula = rula_arm_score(
        ang["shoulder_elevation"], ang["elbow_flexion"],
        ang["shoulder_abduction"],
    )

    return {
        "t": t, "dt": dt, "angles": ang, "rula": rula,
        "op_speed": op_speed, "rob_speed": rob_speed,
        "engaged": engaged, "fidelity": fid, "bandwidth": bw,
        "smoothness": smooth, "stress": stress,
        "wrist": wrist, "arm": arm, "hand": hand, "tcp": tcp,
        "duration_s": float(t[-1] - t[0]) if t.size else 0.0,
        "engaged_s": float(np.sum(engaged) * dt),
    }


def _verdict(a):
    """Short plain statements, each tied to a number in the report."""
    out = []
    f = a["fidelity"]
    if np.isfinite(f["lag_s"]):
        lag_ms = f["lag_s"] * 1000
        out.append((
            f"Transmission lag {lag_ms:.0f} ms",
            OK if lag_ms < 150 else (WARN if lag_ms < 300 else BAD),
            "Under about 150 ms feels direct; beyond 300 ms the operator "
            "starts correcting for the delay rather than the task.",
        ))
    if np.isfinite(f["correlation"]):
        out.append((
            f"Motion correlation {f['correlation']:.2f}",
            OK if f["correlation"] > 0.8 else
            (WARN if f["correlation"] > 0.5 else BAD),
            "How much of what the operator did appears at the robot at all.",
        ))
    b = a["bandwidth"]["bandwidth_hz"]
    if np.isfinite(b):
        out.append((
            f"Usable bandwidth {b:.1f} Hz",
            OK if b > 2.0 else (WARN if b > 1.0 else BAD),
            "Deliberate reaching lives below 2 Hz; fine corrections and "
            "recovery from a slip need more.",
        ))
    r = a["smoothness"]["ratio"]
    if np.isfinite(r):
        out.append((
            f"Smoothness ratio {r:.2f}",
            OK if 0.5 <= r <= 2.0 else WARN,
            "Above 1 the link added jerk the operator did not produce; well "
            "below 1 it filtered the motion, which costs lag.",
        ))
    s = a["stress"]["shoulder_elevation"]
    out.append((
        f"Shoulder above comfort {s['fraction']*100:.0f}% of the session",
        OK if s["fraction"] < 0.1 else
        (WARN if s["fraction"] < 0.3 else BAD),
        f"Median {s['median']:.0f}°, 95th percentile {s['p95']:.0f}°. "
        "Sustained elevation past 60° is the usual source of shoulder "
        "fatigue in teleoperation.",
    ))
    h = a["stress"]["static"]
    out.append((
        f"Held still {h['fraction']*100:.0f}% of the time, "
        f"longest {h['longest_hold_s']:.0f} s",
        OK if h["longest_hold_s"] < 20 else
        (WARN if h["longest_hold_s"] < 60 else BAD),
        "Static holding is invisible in a range-of-motion summary - the "
        "angle is simply constant - but it is what fatigues an operator.",
    ))
    return out


def build_report(a, out_path, title="Teleoperation session"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = a["t"]
    ang = a["angles"]
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="#0e1116",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", size=12),
        margin=dict(l=60, r=30, t=50, b=45),
    )

    figs = []

    # 1. Input against output, the core of the question
    f1 = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Operator wrist speed vs robot tool speed",
                        "Clutch"),
        row_heights=[0.75, 0.25],
    )
    f1.add_trace(go.Scatter(x=t, y=a["op_speed"], name="operator",
                            line=dict(color=ACCENT, width=1.4)), row=1, col=1)
    scale = (np.nanmax(a["op_speed"]) /
             max(np.nanmax(a["rob_speed"]), 1e-9))
    f1.add_trace(go.Scatter(x=t, y=a["rob_speed"] * scale,
                            name=f"robot (x{scale:.1f})",
                            line=dict(color=WARN, width=1.2)), row=1, col=1)
    f1.add_trace(go.Scatter(x=t, y=a["engaged"].astype(float), name="clutch",
                            fill="tozeroy", line=dict(color=OK, width=0)),
                 row=2, col=1)
    f1.update_yaxes(title_text="m/s (robot scaled)", row=1, col=1)
    f1.update_xaxes(title_text="time (s)", row=2, col=1)
    f1.update_layout(height=430, title="Transmission", **layout)
    figs.append(("Is the operator's motion reaching the robot?", f1,
                 "The robot trace is rescaled so the shapes can be compared; "
                 "the actual gain is reported above. What matters here is "
                 "whether the shape follows, and by how much it is delayed."))

    # 2. Frequency response
    b = a["bandwidth"]
    if b["f"].size:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=b["f"], y=b["gain_db"], name="gain",
                                line=dict(color=ACCENT, width=2)))
        f2.add_hline(y=-3, line=dict(color=DIM, dash="dot"),
                     annotation_text="-3 dB")
        if np.isfinite(b["bandwidth_hz"]):
            f2.add_vline(x=b["bandwidth_hz"], line=dict(color=WARN, dash="dash"),
                         annotation_text=f"{b['bandwidth_hz']:.1f} Hz")
        f2.add_trace(go.Scatter(x=b["f"], y=b["coherence"] * 20 - 40,
                                name="coherence (scaled)",
                                line=dict(color=DIM, width=1, dash="dot")))
        f2.update_xaxes(title_text="frequency (Hz)", type="log")
        f2.update_yaxes(title_text="gain (dB, relative to DC)")
        f2.update_layout(height=380, title="Frequency response", **layout)
        figs.append(("What survives the link", f2,
                     "Gain relative to slow motion. The -3 dB crossing is the "
                     "fastest motion the operator can still command. Coherence "
                     "is drawn only as a confidence measure - it stays near 1 "
                     "for any linear filter however much it attenuates, so it "
                     "cannot be used as the bandwidth itself."))

    # 3. Posture over time against the comfort bands
    f3 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       vertical_spacing=0.06,
                       subplot_titles=tuple(
                           k.replace("_", " ") for k in COMFORT))
    for i, key in enumerate(COMFORT, start=1):
        lo, hi = COMFORT[key]
        f3.add_trace(go.Scatter(x=t, y=ang[key], name=key,
                                line=dict(color=ACCENT, width=1.2),
                                showlegend=False), row=i, col=1)
        f3.add_hrect(y0=lo, y1=hi, fillcolor=OK, opacity=0.10,
                     line_width=0, row=i, col=1)
        f3.update_yaxes(title_text="deg", row=i, col=1)
    f3.update_xaxes(title_text="time (s)", row=3, col=1)
    f3.update_layout(height=620, title="Operator posture", **layout)
    figs.append(("What the operator paid", f3,
                 "Shaded bands are the low-risk ranges used in ergonomic "
                 "screening, not anatomical limits. Time spent outside them "
                 "is what accumulates into fatigue."))

    # 4. Where the time was actually spent
    f4 = make_subplots(rows=1, cols=3,
                       subplot_titles=tuple(k.replace("_", " ")
                                            for k in COMFORT))
    for i, key in enumerate(COMFORT, start=1):
        lo, hi = COMFORT[key]
        x = ang[key][np.isfinite(ang[key])]
        f4.add_trace(go.Histogram(x=x, nbinsx=40, marker_color=ACCENT,
                                  showlegend=False), row=1, col=i)
        f4.add_vrect(x0=lo, x1=hi, fillcolor=OK, opacity=0.12,
                     line_width=0, row=1, col=i)
        f4.update_xaxes(title_text="deg", row=1, col=i)
    f4.update_layout(height=320, title="Posture distribution", **layout)
    figs.append(("Where the time went", f4,
                 "A median inside the band with a long tail outside it is a "
                 "different problem from a median sitting on the edge: the "
                 "first is occasional reaching, the second is a workspace "
                 "that needs moving."))

    # 5. Posture risk over time
    f5 = go.Figure()
    f5.add_trace(go.Scatter(x=t, y=a["rula"], name="score",
                            line=dict(color=WARN, width=1.4),
                            fill="tozeroy"))
    f5.update_yaxes(title_text="arm posture score (1 best)", range=[0, 6.5])
    f5.update_xaxes(title_text="time (s)")
    f5.update_layout(height=300, title="Posture risk", **layout)
    figs.append(("Posture risk over time", f5,
                 "A coarse RULA-style arm score, not a validated assessment - "
                 "that needs wrist, neck, trunk and load scored by a trained "
                 "observer. It is here to show when a session drifts into "
                 "territory a real assessment would care about."))

    # ------------------------------------------------------------------ HTML
    import plotly.io as pio

    cards = "".join(
        f'<div class="card" style="border-left-color:{col}">'
        f'<div class="v">{label}</div><div class="n">{note}</div></div>'
        for label, col, note in _verdict(a)
    )
    fid = a["fidelity"]
    summary = (
        f"{a['duration_s']:.0f} s recorded, "
        f"{a['engaged_s']:.0f} s with the clutch engaged &nbsp;·&nbsp; "
        f"lag {fid['lag_s']*1000:.0f} ms &nbsp;·&nbsp; "
        f"gain {fid['gain']:.2f} &nbsp;·&nbsp; "
        f"residual {fid['rmse']:.3f} (operator units)"
    )

    body = []
    for i, (heading, fig, note) in enumerate(figs):
        html = pio.to_html(fig, include_plotlyjs=(i == 0), full_html=False)
        body.append(
            f'<section><h2>{heading}</h2><p class="note">{note}</p>{html}</section>'
        )

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ background:#0e1116; color:#e6edf3; margin:0 auto; max-width:1180px;
        padding:40px 24px 80px;
        font:14px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,sans-serif; }}
 h1 {{ font-size:22px; font-weight:600; margin:0 0 6px; }}
 h2 {{ font-size:16px; font-weight:600; margin:38px 0 4px; }}
 .sub {{ color:#8b949e; margin:0 0 26px; }}
 .note {{ color:#8b949e; margin:0 0 12px; max-width:80ch; }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr));
          gap:12px; margin:22px 0 10px; }}
 .card {{ background:#161b22; border-left:3px solid {DIM}; border-radius:4px;
         padding:12px 14px; }}
 .card .v {{ font-weight:600; margin-bottom:3px; }}
 .card .n {{ color:#8b949e; font-size:12.5px; }}
 section {{ margin-bottom:10px; }}
 footer {{ color:#8b949e; font-size:12.5px; margin-top:50px;
           border-top:1px solid #21262d; padding-top:16px; max-width:80ch; }}
</style></head><body>
<h1>{title}</h1><p class="sub">{summary}</p>
<div class="cards">{cards}</div>
{''.join(body)}
<footer>
Fidelity is measured between the operator's wrist speed and the robot's joint
speed, so it does not depend on the hand-eye transform being correct. Lag and
gain are estimated first and removed before the residual is computed: both are
correctable, and what is left is the part of the motion the link genuinely
failed to convey.<br><br>
The posture bands are ergonomic screening ranges, not anatomical limits, and
the arm score is a coarse flag rather than a validated RULA assessment.
</footer>
</body></html>"""

    with open(out_path, "w") as fh:
        fh.write(doc)
    return out_path


def main(args=None):
    """ROS entry point; also usable as a plain script."""
    argv = sys.argv[1:]
    if argv and argv[0].endswith(".npz"):
        session = argv[0]
        out = argv[1] if len(argv) > 1 else session.replace(".npz", ".html")
        print(build_report(analyse(session), out))
        return

    import rclpy
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node("session_report_node")
    node.declare_parameter("session", "/ws/vendor/session.npz")
    node.declare_parameter("out", "")
    session = node.get_parameter("session").get_parameter_value().string_value
    out = node.get_parameter("out").get_parameter_value().string_value
    out = out or session.replace(".npz", ".html")
    try:
        path = build_report(analyse(session), out)
        node.get_logger().info(f"wrote {path}")
    except Exception as exc:
        node.get_logger().error(f"report failed: {exc}")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
