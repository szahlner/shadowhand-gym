import os
import numpy as np

# TODO: Remove this import
import pybullet as p

from shadowhand_gym.envs.core import Task, get_data_path
from shadowhand_gym.envs import rotations
from shadowhand_gym.pybullet import PyBullet
from shadowhand_gym.envs.robots.shadowhand import ShadowHand


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance to goal.

    Args:
        a (np.ndarray): Achieved goal (position, orientation).
        b (np.ndarray): Desired goal (position, orientation).

    Returns:
        float: Distance to goal.
    """
    assert a.shape == b.shape, "Shape of 'a' must match shape of 'b'"

    # We do not care about the position
    if len(a) == 7:
        a = a[3:]
        b = b[3:]

        a = np.array(p.getEulerFromQuaternion(a))
        b = np.array(p.getEulerFromQuaternion(b))

        a[1] = b[1]
    else:
        a = a[:, 3:]
        b = b[:, 3:]

        x, y = [], []
        for n in range(len(a)):
            x.append(p.getEulerFromQuaternion(a[n]))
            y.append(p.getEulerFromQuaternion(b[n]))
        a = np.array(x)
        b = np.array(y)

        a[:, 1] = b[: , 1]

    a = rotations.euler2quat(a)
    b = rotations.euler2quat(b)

    quaternion_diff = rotations.quat_mul(a, rotations.quat_conjugate(b))
    angle_diff = 2 * np.arccos(np.clip(quaternion_diff[..., 0], -1.0, 1.0))

    return angle_diff


class Block(Task):
    GOAL_ORIENTATION = [0.0] * 3

    def __init__(
        self,
        sim: PyBullet,
        robot: ShadowHand,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        block_half_extend: float = 0.02,
    ) -> None:
        """Shadow dexterous hand manipulate block task.

        Args:
            sim (PyBullet): PyBullet client to interact with the simulator.
            robot (ShadowHand): Shadow dexterous hand robot.
            reward_type (str, optional): Reward type. Choose from 'dense' or 'sparse'. Defaults to 'sparse'.
            distance_threshold (float, optional): Distance threshold to determine between success and failure.
            block_half_extend (float, optional): Half extend of the block in metres. Defaults to 0.02.
        """
        assert reward_type in [
            "dense",
            "sparse",
        ], "Reward type must be 'dense' or 'sparse'"

        self.sim = sim
        self.robot = robot
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.block_half_extend = block_half_extend

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(
                target_position=[0.25, 0.0, 0.0], distance=0.5, yaw=45, pitch=-40
            )

        self.goal = None

    def _create_scene(self) -> None:
        """Create scene.

        Add (ghost) objects and stuff that is not included in the robot URDF file."""
        self.sim.create_object(
            body_name="object",
            mass=0.5,
            object_path=os.path.join(get_data_path(), "assets", "obj", "block.obj"),
            texture_path=os.path.join(
                get_data_path(), "assets", "materials", "textures", "block_2022.png"
            ),
            mesh_scale=[self.block_half_extend] * 3,
            friction=5,  # Increase friction
        )

    def get_goal(self) -> np.ndarray:
        """Return goal.

        Returns:
            np.ndarray: Cartesian position and orientation in quaternions of the object (target).
        """
        return self.goal.copy()

    def get_obs(self) -> np.ndarray:
        """Return task specific observations.

        Returns:
            np.ndarray: Object velocity, angular velocity, cartesian position and orientation in quaternions.
        """
        object_velocity = self.get_object_velocity()
        object_angular_velocity = self.get_object_angular_velocity()
        achieved_goal = self.get_achieved_goal()

        observations = np.concatenate(
            [object_velocity, object_angular_velocity, achieved_goal]
        )
        return observations

    def get_achieved_goal(self) -> np.ndarray:
        """Return achieved goal.

        Returns:
            np.ndarray: Cartesian position and orientation in quaternions.
        """
        object_position = self.get_object_position()
        object_orientation = self.get_object_orientation()

        achieved_goal = np.concatenate([object_position, object_orientation])
        return achieved_goal

    def reset(self) -> None:
        """Reset task."""
        self.goal = self._sample_goal()

        # Let object spawn slightly above the hand palm position to prevent it to fall off
        palm_position = self.robot.get_palm_position()
        object_start_position = palm_position + np.array([0.05, 0.0, 0.075])
        angle = self.np_random.uniform(-np.pi, np.pi)
        axis = self.np_random.uniform(-1.0, 1.0, size=3)
        object_start_orientation = self.sim.physics_client.getQuaternionFromAxisAngle(
            axis, angle
        )

        self.sim.set_base_pose(
            "object", object_start_position, object_start_orientation
        )

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal.

        Returns:
            np.ndarray: Cartesian position and orientation in quaternions of the object (target).
        """
        goal_orientation = self.sim.physics_client.getQuaternionFromEuler(
            self.GOAL_ORIENTATION
        )
        goal_orientation = np.array(goal_orientation)

        object_position = self.get_object_position()

        goal = np.concatenate([object_position, goal_orientation])
        return goal

    def get_object_position(self) -> np.ndarray:
        """Get current object position.

        Returns:
            np.ndarray: Cartesian position of the object.
        """
        position = self.sim.get_base_position("object")
        return np.array(position)

    def get_object_orientation(self) -> np.ndarray:
        """Get current object orientation.

        Returns:
            np.ndarray: Object orientation in quaternions.
        """
        orientation = self.sim.get_base_orientation("object")
        return np.array(orientation)

    def get_object_velocity(self) -> np.ndarray:
        """Get current object cartesian velocity.

        Returns:
            np.ndarray: Cartesian velocity of the object.
        """
        velocity = self.sim.get_base_velocity("object")
        return np.array(velocity)

    def get_object_angular_velocity(self) -> np.ndarray:
        """Get current object angular velocity.

        Returns:
            np.ndarray: Angular velocity of the object.
        """
        velocity = self.sim.get_base_angular_velocity("object")
        return np.array(velocity)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """Return success or failure.

        Returns:
            float: Success or failure (1.0 = success, 0.0 = failure).
        """
        d = distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """Return reward.

        Returns:
            float: The reward for a particular action.
        """
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return (d < self.distance_threshold).astype(np.float32) - 1.0
        else:
            return -d
