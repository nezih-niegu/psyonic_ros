"""Persistent teleop calibration.

Calibration is per-user and per-setup: the retargeting ranges depend on hand
size and camera distance, and the MSJ reference depends on the sample rate,
filter tuning and what that particular user's smooth motion looks like.

Nothing here is guessable from defaults, so the teleop node refuses to command
a hand until it has either loaded a saved calibration or measured a fresh one.

Stored as JSON so it can be inspected and edited by hand.
"""

import json
import os
import time

import numpy as np


def _median(xs):
    return float(np.median(np.asarray(list(xs), dtype=float)))
from pathlib import Path

SCHEMA_VERSION = 1

DEFAULT_PATH = os.path.join(
    os.path.expanduser("~"), ".ros", "ah_teleop_calibration.json"
)

# Keys that must be present for a calibration to be considered usable
REQUIRED_KEYS = [
    "curl_open",
    "curl_closed",
    "thumb_flex_open",
    "thumb_flex_closed",
    "thumb_opp_open",
    "thumb_opp_closed",
    # Limb geometry is required. Without it the body fit solves for scale on
    # every frame from a handful of points, which is under-constrained and
    # drives the scale to its clamp.
    "body_scale",
    "upper_arm_m",
    "forearm_m",
]

OPTIONAL_KEYS = ["msj_reference", "msj_spread", "msj_samples", "limb_samples"]


class CalibrationError(Exception):
    pass


def save_calibration(values, path=None, extra=None):
    """Write calibration to JSON. Returns the path written."""
    path = Path(path or DEFAULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload.update({k: float(values[k]) for k in REQUIRED_KEYS if k in values})
    for k in OPTIONAL_KEYS:
        if k in values and values[k] is not None:
            payload[k] = float(values[k])
    if extra:
        payload["notes"] = extra

    path.write_text(json.dumps(payload, indent=2) + "\n")
    return str(path)


def load_calibration(path=None):
    """Load calibration from JSON, or return None if absent.

    Raises CalibrationError if the file exists but is unusable, because
    silently falling back to defaults would command the hand with the wrong
    ranges.
    """
    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"{path} is not valid JSON: {exc}") from exc

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise CalibrationError(
            f"{path} is missing {missing}. Calibrations saved before the limb "
            "step existed lack the body geometry; run the calibration again."
        )

    if data.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationError(
            f"{path} has schema_version {data.get('schema_version')}, "
            f"expected {SCHEMA_VERSION}. Delete it and recalibrate."
        )

    # Ranges must be non-degenerate or the mapping divides by ~zero
    if abs(data["curl_closed"] - data["curl_open"]) < 1.0:
        raise CalibrationError(
            "curl_open and curl_closed are nearly equal - the calibration "
            "poses were probably captured without moving. Recalibrate."
        )
    return data


def describe(data):
    """One-line human summary of a loaded calibration."""
    if not data:
        return "none"
    ref = data.get("msj_reference")
    spread = data.get("msj_spread")
    return (
        f"curl {data['curl_open']:.1f}..{data['curl_closed']:.1f}, "
        f"thumb flex {data['thumb_flex_open']:.1f}..{data['thumb_flex_closed']:.1f}, "
        f"thumb opp {data['thumb_opp_open']:.1f}..{data['thumb_opp_closed']:.1f}"
        + (
            f", MSJ ref {ref:.1f}"
            + (f" +/- {spread:.1f}" if spread is not None else "")
            if ref is not None
            else ", MSJ ref unset"
        )
        + f" (saved {data.get('saved_at', 'unknown')})"
    )


class CalibrationSession:
    """Guided three-step calibration state machine.

    Steps, in order:
        open     - flat hand, fingers straight, thumb spread
        closed   - fist, thumb across the palm
        baseline - move naturally for a few seconds; the observed MSJ becomes
                   the reference that later motion is scored against

    The node holds one of these and feeds it samples; it does not publish hand
    commands while a session is active.
    """

    STEPS = ["open", "closed", "baseline", "limb"]

    PROMPTS = {
        "open": "Hold a FLAT hand, fingers straight, thumb spread. Press 'o'.",
        "closed": "Make a FIST, thumb across the palm. Press 'c'.",
        "baseline": (
            "Move naturally at your intended teleop speed for a few seconds, "
            "then press 'b'. This motion becomes 'good' (green)."
        ),
        "limb": (
            "Face the camera, arm relaxed and clearly visible. Hold a few "
            "seconds, then press 'l' to measure your arm."
        ),
    }

    def __init__(self):
        self.values = {}
        self.step_index = 0
        self.done = False
        self._baseline_samples = []
        self._limb_samples = []

    @property
    def step(self):
        if self.step_index >= len(self.STEPS):
            return None
        return self.STEPS[self.step_index]

    def prompt(self):
        s = self.step
        return self.PROMPTS[s] if s else "Calibration complete."

    def capture_open(self, features):
        self.values["curl_open"] = float(sum(features[:4]) / 4.0)
        self.values["thumb_flex_open"] = float(features[4])
        self.values["thumb_opp_open"] = float(features[5])
        self._advance("open")

    def capture_closed(self, features):
        self.values["curl_closed"] = float(sum(features[:4]) / 4.0)
        self.values["thumb_flex_closed"] = float(features[4])
        self.values["thumb_opp_closed"] = float(features[5])
        self._advance("closed")

    def observe_baseline(self, msj_total):
        """Accumulate an MSJ sample during the baseline step.

        The node feeds every frame here while the step is active, so the
        reference is the average of real motion rather than one instant, and
        the spread of that motion is measured too.
        """
        if msj_total > 0.0:
            self._baseline_samples.append(float(msj_total))

    @property
    def baseline_count(self):
        return len(self._baseline_samples)

    def capture_baseline(self, msj_total=None):
        """Finish the baseline step, setting the reference and its spread."""
        samples = list(self._baseline_samples)
        if not samples and msj_total:
            samples = [float(msj_total)]
        if not samples:
            raise CalibrationError("no MSJ samples collected")

        mean = sum(samples) / len(samples)
        var = sum((s - mean) ** 2 for s in samples) / max(len(samples) - 1, 1)
        self.values["msj_reference"] = float(mean)
        self.values["msj_spread"] = float(var ** 0.5)
        self.values["msj_samples"] = len(samples)
        self._advance("baseline")

    def observe_limb(self, scale, upper_arm, forearm):
        """Accumulate a body-scale and segment-length sample."""
        if scale and np.isfinite(scale) and 0.3 < scale < 2.0:
            self._limb_samples.append((float(scale), float(upper_arm),
                                       float(forearm)))

    @property
    def limb_count(self):
        return len(self._limb_samples)

    def capture_limb(self):
        """Finish the limb step using the median of what was observed.

        Median, not mean: a few frames will have fitted badly, and one wild
        sample would drag a mean well off.
        """
        if not self._limb_samples:
            raise CalibrationError("no limb samples collected")
        arr = list(zip(*self._limb_samples))
        self.values["body_scale"] = float(_median(arr[0]))
        self.values["upper_arm_m"] = float(_median(arr[1]))
        self.values["forearm_m"] = float(_median(arr[2]))
        self.values["limb_samples"] = len(self._limb_samples)
        self._advance("limb")

    def _advance(self, from_step):
        if self.step == from_step:
            self.step_index += 1
        if self.step_index >= len(self.STEPS):
            self.done = True

    def validate(self):
        """Return a list of problems, empty if the calibration is sane."""
        problems = []
        v = self.values
        if "curl_open" in v and "curl_closed" in v:
            if v["curl_closed"] - v["curl_open"] < 20.0:
                problems.append(
                    "open and closed curl are too close together - hold the "
                    "poses more distinctly"
                )
        if "thumb_opp_open" in v and "thumb_opp_closed" in v:
            if abs(v["thumb_opp_open"] - v["thumb_opp_closed"]) < 5.0:
                problems.append("thumb opposition barely changed between poses")
        return problems
