ARG ROS_DISTRO=humble
FROM osrf/ros:${ROS_DISTRO}-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /ws

# ---------------------------------------------------------------- apt packages
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-pip \
    x11-apps \
    curl \
    ca-certificates \
    gnupg \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-xacro \
    vim \
    tmux \
    # --- needed by xarm_api and the xarm launch files ---
    ros-${ROS_DISTRO}-control-msgs \
    ros-${ROS_DISTRO}-launch-param-builder \
    ros-${ROS_DISTRO}-moveit-configs-utils \
    ros-${ROS_DISTRO}-srdfdom \
    # --- OpenGL for the MuJoCo viewer and the clinical UI ---
    libgl1 \
    libglew2.2 \
    libosmesa6 \
    libglfw3 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------- K4A SDK
# Not in the ROS or Ubuntu repos. Microsoft publishes for 18.04 (bionic) only;
# the packages install and work on 22.04. libk4a1.4-dev provides
# k4aConfig.cmake, without which azure_kinect_ros_driver fails to configure.
#
# The EULA prompt cannot be answered during a build, so it is preseeded.
RUN curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/ubuntu/18.04/prod bionic main" \
        > /etc/apt/sources.list.d/microsoft-prod.list && \
    echo 'libk4a1.4 libk4a1.4/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' \
        | debconf-set-selections && \
    echo 'libk4a1.4 libk4a1.4/accept-eula boolean true' \
        | debconf-set-selections && \
    apt-get update && apt-get install -y --no-install-recommends \
        libk4a1.4 libk4a1.4-dev k4a-tools \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------ python packages
COPY ./requirements /tmp/requirements
RUN pip install --no-cache-dir -r /tmp/requirements/requirements.txt

# ---------------------------------------------------------------- entrypoint
COPY ./docker/entrypoint.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/tmp/entrypoint.sh"]
CMD ["bash"]
