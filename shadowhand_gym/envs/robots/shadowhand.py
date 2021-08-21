import os
from abc import ABC

import numpy as np

from gym import spaces

from shadowhand_gym.envs.core import PyBulletRobot, get_data_path
from shadowhand_gym.pybullet import PyBullet


from typing import List, Optional


class ShadowHand(PyBulletRobot, ABC):

    JOINT_INDICES = [
        1,
        2,  # b"rh_WRJ2", b"rh_WRJ1"
        3,
        4,
        5,
        6,  # b"rh_FFJ4", b"rh_FFJ3", b"rh_FFJ2", b"rh_FFJ1"
        8,
        9,
        10,
        11,  # b"rh_MFJ4", b"rh_MFJ3", b"rh_MFJ2", b"rh_MFJ1"
        13,
        14,
        15,
        16,  # b"rh_RFJ4", b"rh_RFJ3", b"rh_RFJ2", b"rh_RFJ1"
        18,
        19,
        20,
        21,
        22,  # b"rh_LFJ5", b"rh_LFJ4", b"rh_LFJ3", b"rh_LFJ2", b"rh_LFJ1"
        24,
        25,
        26,
        27,
        28,  # b"rh_THJ5", b"rh_THJ4", b"rh_THJ3", b"rh_THJ2", b"rh_THJ1"
    ]

    COUPLED_JOINTS = {
        6: 5,  # b"rh_FFJ1": b"rh_FFJ2"
        11: 10,  # b"rh_MFJ1": b"rh_MFJ2"
        16: 15,  # b"rh_RFJ1": b"rh_RFJ2"
        22: 21,  # b"rh_LFJ1": b"rh_LFJ2"
    }

    JOINT_FORCES = [
        150,
        150,  # b"rh_WRJ2", b"rh_WRJ1"
        150,
        150,
        150,
        150,  # b"rh_FFJ4", b"rh_FFJ3", b"rh_FFJ2", b"rh_FFJ1"
        150,
        150,
        150,
        150,  # b"rh_MFJ4", b"rh_MFJ3", b"rh_MFJ2", b"rh_MFJ1"
        150,
        150,
        150,
        150,  # b"rh_RFJ4", b"rh_RFJ3", b"rh_RFJ2", b"rh_RFJ1"
        150,
        150,
        150,
        150,
        150,  # b"rh_LFJ5", b"rh_LFJ4", b"rh_LFJ3", b"rh_LFJ2", b"rh_LFJ1"
        150,
        150,
        150,
        150,
        150,  # b"rh_THJ5", b"rh_THJ4", b"rh_THJ3", b"rh_THJ2", b"rh_THJ1"
    ]

    NEUTRAL_JOINT_VALUES = [
        -0.06,
        0.09,  # b"rh_WRJ2", b"rh_WRJ1"
        -0.06,
        0.0,
        0.53,
        0.53,  # b"rh_FFJ4", b"rh_FFJ3", b"rh_FFJ2", b"rh_FFJ1"
        0.02,
        0.0,
        0.54,
        0.54,  # b"rh_MFJ4", b"rh_MFJ3", b"rh_MFJ2", b"rh_MFJ1"
        -0.06,
        0.0,
        0.54,
        0.54,  # b"rh_RFJ4", b"rh_RFJ3", b"rh_RFJ2", b"rh_RFJ1"
        0.0,
        -0.22,
        0.0,
        0.54,
        0.54,  # b"rh_LFJ5", b"rh_LFJ4", b"rh_LFJ3", b"rh_LFJ2", b"rh_LFJ1"
        1.05,
        0.49,
        0.21,
        -0.02,
        0.28,  # b"rh_THJ5", b"rh_THJ4", b"rh_THJ3", b"rh_THJ2", b"rh_THJ1"
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

    def get_obs(self):
        positions = self.get_positions()
        velocities = self.get_velocities()
        obs = np.concatenate([positions, velocities])

        return obs

    def get_positions(self):
        """Returns the joint positions."""
        positions = []

        for joint in self.JOINT_INDICES:
            positions.append(self.get_joint_position(joint))

        return np.array(positions)

    def get_velocities(self):
        """Returns the joint velocities."""
        velocities = []

        for joint in self.JOINT_INDICES:
            velocities.append(self.get_joint_velocity(joint))

        return np.array(velocities)

    def get_fingertip_positions(self):
        positions = []

        for joint in self.FINGERTIP_LINKS:
            positions.append(self.get_link_position(joint))

        return np.array(positions)

    def get_palm_position(self):
        position = self.get_link_position(self.PALM_LINK)

        return np.array(position)

    def reset(self):
        self.set_joint_neutral()

    def set_joint_neutral(self):
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
