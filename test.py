import math
import time

import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data


# Global simulation settings requested by the spec.
SIM_DURATION = 10.0
DT = 0.01
JOINT_NAMES = ["joint0", "joint1", "joint2"]
PAYLOAD_LINK_NAME = "payload"

# q(t) = A*sin(w*t), with A = pi/2 (±90 deg).
# Choose w so max |qdot| = A*w = 1 rad/s.
AMP_RAD = math.pi / 10.0
OMEGA = 1.0 / AMP_RAD

# Position-control settings (tune if tracking looks too soft/stiff).
POSITION_GAINS = [0.18, 0.16, 0.14]
VELOCITY_GAINS = [0.45, 0.40, 0.35]
MAX_FORCES = [800.0, 800.0, 800.0]


def plot_results(time_log, torque_log, vel_log, acc_log, target_pos_log, actual_pos_log, payload_speed_log):
    # Plot torque, velocity, and acceleration for joint0/1/2.
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    for i, name in enumerate(JOINT_NAMES):
        torque_line, = axes[0].plot(time_log, torque_log[i], label=name)
        axes[1].plot(time_log, vel_log[i], label=name)
        acc_line, = axes[2].plot(time_log, acc_log[i], label=name)

        avg_torque = sum(abs(v) for v in torque_log[i]) / len(torque_log[i])
        avg_acc = sum(abs(v) for v in acc_log[i]) / len(acc_log[i])
        axes[0].axhline(
            avg_torque,
            linestyle=":",
            color=torque_line.get_color(),
            linewidth=1.8,
            label=f"{name} avg",
        )
        axes[2].axhline(
            avg_acc,
            linestyle=":",
            color=acc_line.get_color(),
            linewidth=1.8,
            label=f"{name} avg",
        )

    axes[0].set_ylabel("Torque [Nm]")
    axes[0].set_xlabel("Time [s]")
    axes[1].set_ylabel("Ang. velocity [rad/s]")
    axes[1].set_xlabel("Time [s]")
    axes[2].set_ylabel("Ang. acceleration [rad/s^2]")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    fig.suptitle("Joint responses for sine motion in [-90, +90] deg")
    fig.tight_layout()

    # Optional payload speed figure to compare against 0.5 m/s target.
    plt.figure(figsize=(11, 3.5))
    plt.plot(time_log, payload_speed_log, label="payload speed")
    plt.plot(time_log, [0.5] * len(time_log), "--", label="target 0.5 m/s")
    plt.xlabel("Time [s]")
    plt.ylabel("Payload speed [m/s]")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.title("Payload linear speed")
    plt.tight_layout()

    # Plot target vs actual joint position in separate subplots (one per joint).
    fig_pos, axes_pos = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for i, name in enumerate(JOINT_NAMES):
        axes_pos[i].plot(time_log, target_pos_log[i], "--", label=f"{name} target")
        axes_pos[i].plot(time_log, actual_pos_log[i], label=f"{name} actual")
        axes_pos[i].set_ylabel("Position [rad]")
        axes_pos[i].set_xlabel("Time [s]")
        axes_pos[i].grid(True, alpha=0.3)
        axes_pos[i].legend(loc="upper right")
    fig_pos.suptitle("Target vs actual joint position")
    fig_pos.tight_layout()

    plt.show()


def main():
    # 1) Connect to PyBullet GUI and configure physics.
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSolverIterations=120)

    # 2) Load environment and robot model.
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("3dof.urdf", useFixedBase=True)

    # 3) Build lookup tables from URDF names to indices for robust access.
    joint_name_to_index = {}
    link_name_to_index = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        joint_name = info[1].decode("utf-8")
        link_name = info[12].decode("utf-8")
        joint_name_to_index[joint_name] = j
        link_name_to_index[link_name] = j

    joint_indices = [joint_name_to_index[name] for name in JOINT_NAMES]
    payload_link_index = link_name_to_index[PAYLOAD_LINK_NAME]

    # Initialize joints at start of trajectory to reduce startup transients.
    for ji in joint_indices:
        p.resetJointState(robot_id, ji, targetValue=0.0, targetVelocity=1.0)

    # 4) Initialize logging containers.
    num_steps = int(SIM_DURATION / DT)
    prev_vel = [0.0, 0.0, 0.0]

    time_log = []
    torque_log = [[], [], []]
    vel_log = [[], [], []]
    acc_log = [[], [], []]
    target_pos_log = [[], [], []]
    actual_pos_log = [[], [], []]
    payload_speed_log = []

    # 5) Main simulation loop (10 s at 0.01 s step).
    for step in range(num_steps + 1):
        t_now = step * DT

        # 5a) Generate sinusoidal joint targets in [-90, +90] deg.
        target_pos = AMP_RAD * math.sin(OMEGA * t_now)
        target_vel = AMP_RAD * OMEGA * math.cos(OMEGA * t_now)  # peak 1 rad/s

        # 5b) Command all 3 actuated joints in one API call.
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[target_pos, target_pos, target_pos],
            targetVelocities=[target_vel, target_vel, target_vel],
            forces=MAX_FORCES,
            positionGains=POSITION_GAINS,
            velocityGains=VELOCITY_GAINS,
        )

        p.stepSimulation()

        # 5c) Read motor torque and joint velocity, then estimate acceleration.
        joint_states = p.getJointStates(robot_id, joint_indices)
        joint_pos = [s[0] for s in joint_states]
        joint_vel = [s[1] for s in joint_states]
        joint_torque = [s[3] for s in joint_states]
        joint_acc = [(joint_vel[i] - prev_vel[i]) / DT for i in range(3)]
        prev_vel = joint_vel

        # 5d) Track payload linear speed to compare against 0.5 m/s target.
        payload_state = p.getLinkState(robot_id, payload_link_index, computeLinkVelocity=1)
        vx, vy, vz = payload_state[6]
        payload_speed = math.sqrt(vx * vx + vy * vy + vz * vz)

        # 5e) Log signals for later plotting with matplotlib.
        time_log.append(t_now)
        payload_speed_log.append(payload_speed)
        for i in range(3):
            torque_log[i].append(joint_torque[i])
            vel_log[i].append(joint_vel[i])
            acc_log[i].append(joint_acc[i])
            target_pos_log[i].append(target_pos)
            actual_pos_log[i].append(joint_pos[i])

        time.sleep(DT)

    # 6) Close simulation and display plots.
    p.disconnect(client)
    plot_results(time_log, torque_log, vel_log, acc_log, target_pos_log, actual_pos_log, payload_speed_log)


if __name__ == "__main__":
    main()
