import os
import numpy as np

from shadowhand_gym.envs.core import PyBulletRobot, get_data_path
from shadowhand_gym.pybullet import PyBullet

from typing import List, Optional, Union
from gym import spaces
from abc import ABC


MOVABLE_JOINTS = [
    b"rh_FFJ4",
    b"rh_FFJ3",
    b"rh_FFJ2",
    b"rh_MFJ4",
    b"rh_MFJ3",
    b"rh_MFJ2",
    b"rh_RFJ4",
    b"rh_RFJ3",
    b"rh_RFJ2",
    b"rh_LFJ5",
    b"rh_LFJ4",
    b"rh_LFJ3",
    b"rh_LFJ2",
    b"rh_THJ5",
    b"rh_THJ4",
    b"rh_THJ3",
    b"rh_THJ2",
    b"rh_THJ1",
    b"rh_WRJ2",
    b"rh_WRJ1",
]

COUPLED_JOINTS = {
    b"rh_FFJ1": b"rh_FFJ2",
    b"rh_MFJ1": b"rh_MFJ2",
    b"rh_RFJ1": b"rh_RFJ2",
    b"rh_LFJ1": b"rh_LFJ2",
}

FINGER_TIPS = [b"rh_FFtip", b"rh_MFtip", b"rh_RFtip", b"rh_LFtip", b"rh_thtip"]

INITIAL_POSITIONS = {
    b"rh_WRJ2": -0.05866723135113716,
    b"rh_WRJ1": 0.08598895370960236,
    b"rh_FFJ4": -0.05925952824065458,
    b"rh_FFJ3": 0.0,
    b"rh_FFJ2": 0.5306965075027753,
    b"rh_FFJ1": 0.5306965075027753,
    b"rh_MFJ4": 0.015051404275727428,
    b"rh_MFJ3": 0.0,
    b"rh_MFJ2": 0.5364634589883859,
    b"rh_MFJ1": 0.5364634589883859,
    b"rh_RFJ4": -0.056137955514170744,
    b"rh_RFJ3": 0.0,
    b"rh_RFJ2": 0.5362351077308591,
    b"rh_RFJ1": 0.5362351077308591,
    b"rh_LFJ5": 0.0,
    b"rh_LFJ4": -0.216215152247765,
    b"rh_LFJ3": 0.0,
    b"rh_LFJ2": 0.542813974505131,
    b"rh_LFJ1": 0.542813974505131,
    b"rh_THJ5": 1.047,
    b"rh_THJ4": 0.4912634677627796,
    b"rh_THJ3": 0.209,
    b"rh_THJ2": -0.024347361541391634,
    b"rh_THJ1": 0.28372550178530886,
}


class ShadowHand(PyBulletRobot, ABC):

    PALM_LINK = 2

    JOINTS_LIMIT_LOW: Union[List[float], np.ndarray]
    JOINTS_LIMIT_HIGH: Union[List[float], np.ndarray]
    JOINTS_MOVABLE: List[int]
    JOINTS_COUPLED: List[int]

    FINGERTIP_LINKS: List[int]

    ACT_CENTER: np.ndarray
    ACT_RANGE: np.ndarray

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

        self.JOINTS_LIMIT_LOW, self.JOINTS_LIMIT_HIGH = [], []
        self.JOINTS_MOVABLE, self.JOINTS_COUPLED = [], []
        self.FINGERTIP_LINKS = []

        self.n_joints = self.get_num_joints()
        for n in range(self.n_joints):
            joint_name = self.get_joint_name(joint=n)

            if joint_name in MOVABLE_JOINTS:
                lower_limit, upper_limit = self.get_joint_limits(joint=n)
                self.JOINTS_LIMIT_LOW.append(lower_limit)
                self.JOINTS_LIMIT_HIGH.append(upper_limit)
                self.JOINTS_MOVABLE.append(n)
            elif joint_name in COUPLED_JOINTS:
                self.JOINTS_COUPLED.append(n)
            elif joint_name in FINGER_TIPS:
                # The fingertip actually is a joint in the URDF file
                self.FINGERTIP_LINKS.append(n)

        self.JOINTS_LIMIT_LOW = np.array(self.JOINTS_LIMIT_LOW)
        self.JOINTS_LIMIT_HIGH = np.array(self.JOINTS_LIMIT_HIGH)

        self.ACT_RANGE = (self.JOINTS_LIMIT_HIGH - self.JOINTS_LIMIT_LOW) / 2.0
        self.ACT_CENTER = (self.JOINTS_LIMIT_HIGH + self.JOINTS_LIMIT_LOW) / 2.0

    def set_action(self, action: np.ndarray) -> None:
        """Set action.

        Args:
            action (np.ndarray): Action to be set.
        """
        # Ensure action does not get changed
        action = action.copy().flatten()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Map to real actions [-1.0, 1.0] -> [action.min, action.max]
        ctrl = self.ACT_CENTER + self.ACT_RANGE * action
        ctrl = np.clip(ctrl, self.JOINTS_LIMIT_LOW, self.JOINTS_LIMIT_HIGH)

        # Deal with coupled joints
        # TODO: Make couple_factor editable
        joint_indices = []
        joint_target_positions = []
        for n in range(self.n_joints):
            if n in self.JOINTS_MOVABLE:
                k = self.JOINTS_MOVABLE.index(n)
                joint_target_positions.append(ctrl[k])
            else:
                if n in self.JOINTS_COUPLED and n - 1 in self.JOINTS_MOVABLE:
                    k = self.JOINTS_MOVABLE.index(n - 1)
                    joint_target_positions.append(ctrl[k])
                else:
                    joint_target_positions.append(0.0)
            joint_indices.append(n)

        self.control_joints(
            joint_indices=joint_indices, target_positions=joint_target_positions
        )

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

        for joint in self.JOINTS_MOVABLE:
            positions.append(self.get_joint_position(joint))

        return np.array(positions)

    def get_velocities(self) -> np.ndarray:
        """Returns the joint velocities.

        Returns:
            np.ndarray: Robot joint velocities.
        """
        velocities = []

        for joint in self.JOINTS_MOVABLE:
            velocities.append(self.get_joint_velocity(joint))

        return np.array(velocities)

    def get_fingertip_positions(self) -> np.ndarray:
        """Returns the fingertip positions.

        Returns:
            np.ndarray: Cartesian fingertip positions.
        """
        positions = []

        for link in self.FINGERTIP_LINKS:
            positions.append(self.get_link_position(link))

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

        # Let the neutral pose settle in
        self.sim.step()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        joint_indices, joint_positions = [], []
        for n in range(self.n_joints):
            joint_name = self.get_joint_name(joint=n)

            try:
                joint_positions.append(INITIAL_POSITIONS[joint_name])
            except KeyError:
                joint_positions.append(0.0)

            joint_indices.append(n)

        self.set_joint_positions(joint_indices=joint_indices, positions=joint_positions)
