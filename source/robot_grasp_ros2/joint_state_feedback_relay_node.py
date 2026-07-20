from __future__ import annotations

from contextlib import suppress
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from src.robot.client import RobotArmClientConfig


class JointStateFeedbackRelayNode(Node):
    """Relay Piper feedback joints into RViz-friendly /joint_states names."""

    def __init__(self) -> None:
        super().__init__("joint_state_feedback_relay")

        defaults = RobotArmClientConfig()
        self.declare_parameter("input_topic", defaults.joint_state_topic)
        self.declare_parameter("output_topic", defaults.display_joint_state_topic)
        self.declare_parameter("include_gripper_pair", True)

        self._input_topic = str(self.get_parameter("input_topic").value)
        self._output_topic = str(self.get_parameter("output_topic").value)
        self._include_gripper_pair = bool(self.get_parameter("include_gripper_pair").value)

        self._publisher = self.create_publisher(JointState, self._output_topic, 20)
        self.create_subscription(JointState, self._input_topic, self._on_joint_state, 20)
        self._missing_joint_warned: set[str] = set()

        self.get_logger().info(
            f"Relaying joint feedback {self._input_topic} -> {self._output_topic}"
        )

    def _lookup(self, msg: JointState, name: str) -> Optional[tuple[float, float, float]]:
        if name not in msg.name:
            return None
        index = msg.name.index(name)
        position = msg.position[index] if index < len(msg.position) else 0.0
        velocity = msg.velocity[index] if index < len(msg.velocity) else 0.0
        effort = msg.effort[index] if index < len(msg.effort) else 0.0
        return float(position), float(velocity), float(effort)

    def _on_joint_state(self, msg: JointState) -> None:
        joint_names = [f"joint{i}" for i in range(1, 7)]
        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []

        for joint_name in joint_names:
            state = self._lookup(msg, joint_name)
            if state is None:
                if joint_name not in self._missing_joint_warned:
                    self.get_logger().warning(
                        f"Missing joint '{joint_name}' in feedback; skip frame"
                    )
                    self._missing_joint_warned.add(joint_name)
                return
            self._missing_joint_warned.discard(joint_name)
            position, velocity, effort = state
            positions.append(position)
            velocities.append(velocity)
            efforts.append(effort)

        gripper_names: list[str] = []
        if self._include_gripper_pair:
            gripper_joint1 = self._lookup(msg, "joint7")
            gripper_joint2 = self._lookup(msg, "joint8")
            if gripper_joint1 is not None and gripper_joint2 is not None:
                positions.extend([gripper_joint1[0], gripper_joint2[0]])
                velocities.extend([gripper_joint1[1], gripper_joint2[1]])
                efforts.extend([gripper_joint1[2], gripper_joint2[2]])
                gripper_names = ["gripper_joint1", "gripper_joint2"]
            else:
                gripper = self._lookup(msg, "gripper")
                if gripper is not None:
                    stroke = max(0.0, gripper[0])
                    half_stroke = stroke / 2.0
                    positions.extend([half_stroke, -half_stroke])
                    velocities.extend([0.0, 0.0])
                    efforts.extend([gripper[2] / 2.0, -gripper[2] / 2.0])
                    gripper_names = ["gripper_joint1", "gripper_joint2"]

        joint_names.extend(gripper_names)

        out = JointState()
        out.header = msg.header
        out.name = joint_names
        out.position = positions
        out.velocity = velocities
        out.effort = efforts
        self._publisher.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointStateFeedbackRelayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        with suppress(KeyboardInterrupt):
            node.destroy_node()
        if rclpy.ok():
            with suppress(KeyboardInterrupt):
                rclpy.shutdown()


if __name__ == "__main__":
    main()
