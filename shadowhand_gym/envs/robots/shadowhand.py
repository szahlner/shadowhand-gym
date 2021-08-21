import os
import numpy as np

from shadowhand_gym.envs.core import PyBulletRobot, get_data_path
from shadowhand_gym.pybullet import PyBullet

from typing import List, Optional
from gym import spaces
from abc import ABC


class ShadowHand(PyBulletRobot, ABC):

    JOINT_INDICES = [
        1,  # b"rh_WRJ2"
        2,  # b"rh_WRJ1"
        3,  # b"rh_FFJ4"
        4,  # b"rh_FFJ3"
        5,  # b"rh_FFJ2"
        6,  # b"rh_FFJ1"
        8,  # b"rh_MFJ4"
        9,  # b"rh_MFJ3"
        10,  # b"rh_MFJ2"
        11,  # b"rh_MFJ1"
        13,  # b"rh_RFJ4"
        14,  # b"rh_RFJ3"
        15,  # b"rh_RFJ2"
        16,  # b"rh_RFJ1"
        18,  # b"rh_LFJ5"
        19,  # b"rh_LFJ4"
        20,  # b"rh_LFJ3"
        21,  # b"rh_LFJ2"
        22,  # b"rh_LFJ1"
        24,  # b"rh_THJ5"
        25,  # b"rh_THJ4"
        26,  # b"rh_THJ3"
        27,  # b"rh_THJ2"
        28,  # b"rh_THJ1"
    ]

    COUPLED_JOINTS = {
        6: 5,  # b"rh_FFJ1": b"rh_FFJ2"
        11: 10,  # b"rh_MFJ1": b"rh_MFJ2"
        16: 15,  # b"rh_RFJ1": b"rh_RFJ2"
        22: 21,  # b"rh_LFJ1": b"rh_LFJ2"
    }

    JOINT_FORCES = [
        150,  # b"rh_WRJ2"
        150,  # b"rh_WRJ1"
        150,  # b"rh_FFJ4"
        150,  # b"rh_FFJ3"
        150,  # b"rh_FFJ2"
        150,  # b"rh_FFJ1"
        150,  # b"rh_MFJ4"
        150,  # b"rh_MFJ3"
        150,  # b"rh_MFJ2"
        150,  # b"rh_MFJ1"
        150,  # b"rh_RFJ4"
        150,  # b"rh_RFJ3"
        150,  # b"rh_RFJ2"
        150,  # b"rh_RFJ1"
        150,  # b"rh_LFJ5"
        150,  # b"rh_LFJ4"
        150,  # b"rh_LFJ3"
        150,  # b"rh_LFJ2"
        150,  # b"rh_LFJ1"
        150,  # b"rh_THJ5"
        150,  # b"rh_THJ4"
        150,  # b"rh_THJ3"
        150,  # b"rh_THJ2"
        150,  # b"rh_THJ1"
    ]

    NEUTRAL_JOINT_VALUES = [
        -0.06,  # b"rh_WRJ2"
        0.09,  # b"rh_WRJ1"
        -0.06,  # b"rh_FFJ4"
        0.0,  # b"rh_FFJ3"
        0.53,  # b"rh_FFJ2"
        0.53,  # b"rh_FFJ1"
        0.02,  # b"rh_MFJ4"
        0.0,  # b"rh_MFJ3"
        0.54,  # b"rh_MFJ2"
        0.54,  # b"rh_MFJ1"
        -0.06,  # b"rh_RFJ4"
        0.0,  # b"rh_RFJ3"
        0.54,  # b"rh_RFJ2"
        0.54,  # b"rh_RFJ1"
        0.0,  # b"rh_LFJ5"
        -0.22,  # b"rh_LFJ4"
        0.0,  # b"rh_LFJ3"
        0.54,  # b"rh_LFJ2"
        0.54,  # b"rh_LFJ1"
        1.05,  # b"rh_THJ5"
        0.49,  # b"rh_THJ4"
        0.21,  # b"rh_THJ3"
        -0.02,  # b"rh_THJ2"
        0.28,  # b"rh_THJ1"
    ]

    FINGERTIP_LINKS = [7, 12, 17, 23, 29]

    PALM_LINK = 2

    def __init__(
        self,
        sim: PyBullet,
        base_position: Optional[List[float]] = None,
        base_orientation: Optional[List[float]] = None,
        position_gain: Optional[float] = None,
        finger_friction: Optional[List[float]] = None,
    ) -> None:
        """Shadow dexterous hand robot.

        Args:
            sim (PyBullet): PyBullet client to interact with the simulator.
            base_position (List[float], optional): Cartesian base position of the robot. Defaults to [0.0, 0.0, 0.0].
            base_orientation (List[float], optional): Base orientation of the robot in quaternions.
                Defaults to [0.0, 0.0, 0.0, 1.0]
            position_gain (float, optional): Position gain for the motors (actuators) in the robot. Defaults to 0.02.
            finger_friction (float, optional): Lateral friction of the robot (all parts). Defaults to 1.0.
        """
        if base_position is None:
            base_position = [0.0] * 3

        if base_orientation is None:
            base_orientation = [0.0, 0.0, 0.0, 1.0]

        if finger_friction is None:
            finger_friction = [1.0] * 5

        if position_gain is not None:
            sim.position_gain = position_gain

        assert len(base_position) == 3, "Position must be of length 3: [x, y, z]"
        assert len(finger_friction) == 5, "Finger_friction must be of length 5"

        n_action = 20
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)

        super().__init__(
            sim=sim,
            body_name="shadow_hand",
            file_name=os.path.join(
                get_data_path(), "assets", "urdf", "shadow_hand.urdf"
            ),
            base_position=base_position,
            base_orientation=base_orientation,
        )

        for n in range(len(self.FINGERTIP_LINKS)):
            self.sim.set_friction(
                self.body_name, self.FINGERTIP_LINKS[n], finger_friction[n]
            )

    def set_action(self, action: np.ndarray) -> None:
        """Set action.

        Args:
            action (np.ndarray): Action to be set.
        """
        # Ensure action does not get changed
        action = action.copy()
        action = action.flatten()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.COUPLED_JOINTS is not None:
            keys = self.COUPLED_JOINTS.keys()

            positions = []
            n = 0
            for joint in self.JOINT_INDICES:
                if joint in keys:
                    positions.append(
                        action[self.JOINT_INDICES.index(self.COUPLED_JOINTS[joint])]
                    )
                else:
                    positions.append(action[n])
                    n += 1

            action = positions

        self.control_joints(action)

    def get_obs(self) -> np.ndarray:
        """Return robot specific observations.

        Returns:
            np.ndarray: Robot joint positions and velocities.
        """
        positions = self.get_positions()
        velocities = self.get_velocities()
        obs = np.concatenate([positions, velocities])

        return obs

    def get_positions(self) -> np.ndarray:
        """Returns the joint positions.

        Returns:
            np.ndarray: Robot joint positions.
        """
        positions = []

        for joint in self.JOINT_INDICES:
            positions.append(self.get_joint_position(joint))

        return np.array(positions)

    def get_velocities(self) -> np.ndarray:
        """Returns the joint velocities.

        Returns:
            np.ndarray: Robot joint velocities.
        """
        velocities = []

        for joint in self.JOINT_INDICES:
            velocities.append(self.get_joint_velocity(joint))

        return np.array(velocities)

    def get_fingertip_positions(self) -> np.ndarray:
        """Returns the fingertip positions.

        Returns:
            np.ndarray: Cartesian fingertip positions.
        """
        positions = []

        for joint in self.FINGERTIP_LINKS:
            positions.append(self.get_link_position(joint))

        return np.array(positions)

    def get_palm_position(self) -> np.ndarray:
        """Returns palm position.

        Returns:
            np.ndarray: Cartesian position of the hand palm.
        """
        position = self.get_link_position(self.PALM_LINK)

        return np.array(position)

    def reset(self) -> None:
        """Resets the robot."""
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_values(self.NEUTRAL_JOINT_VALUES)

    def set_joint_values(self, positions: List[float]) -> None:
        """set the joint position/angles of a body.

        Can induce collisions.

        Args:
            positions (List[float]): Joint positions/angles.
        """
        self.sim.set_joint_positions(
            self.body_name, joint_indices=self.JOINT_INDICES, positions=positions
        )
