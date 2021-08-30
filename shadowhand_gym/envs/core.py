import os
import numpy as np
import gym

from shadowhand_gym.pybullet import PyBullet

from gym import utils, spaces
from typing import List, Optional, Tuple, Union
from abc import ABC


def get_data_path() -> str:
    """Return the absolute data path.

    Returns:
        str: Absolute data path.
    """
    data_path = os.path.join(os.path.dirname(__file__))
    return data_path


class PyBulletRobot:

    JOINT_INDICES = None
    JOINT_FORCES = None

    def __init__(
        self,
        sim: PyBullet,
        body_name: str,
        file_name: str,
        base_position: List[float],
        base_orientation: List[float],
    ) -> None:
        """Base class for robot env.

        Args:
            sim (PyBullet): The simulation engine.
            body_name (str): The name of the robot within the simulation.
            file_name (str): Path of the URDF file.
            base_position (List[float]): Cartesian position of the base of the robot [x, y, z].
            base_orientation (List[float]): Orientation of the base of the robot in quaternions [x, y, z, w].
        """
        self.sim = sim
        self.body_name = body_name
        with self.sim.no_rendering():
            self._load_robot(
                file_name=file_name,
                base_position=base_position,
                base_orientation=base_orientation,
            )
            self.setup()

    def _load_robot(
        self, file_name: str, base_position: List[float], base_orientation: List[float]
    ) -> None:
        """Load the robot.

        Args:
            file_name (str): Path of the URDF file.
            base_position (List[float]): Cartesian position of the base of the robot [x, y, z].
                Defaults to [0.0, 0.0, 0.0].
            base_orientation (List[float]): Orientation of the base of the robot in quaternions [x, y, z, w].
                Defaults to [0.0, 0.0, 0.0, 1.0].
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=True,
        )

    def setup(self):
        """Called once in the constructor."""
        pass

    def set_action(self, action):
        """Perform the action."""
        raise NotImplementedError

    def get_obs(self):
        """Return the observation associated to the robot."""
        raise NotImplementedError

    def reset(self):
        """Reset the robot."""
        raise NotImplementedError

    def get_joint_position(self, joint: int) -> float:
        """Returns the position of the joint.

        Args:
            joint (int): Joint index in the body.

        Returns:
            float: Joint position.
        """
        return self.sim.get_joint_position(self.body_name, joint)

    def get_joint_velocity(self, joint: int) -> float:
        """Returns the velocity of the joint.

        Args:
            joint (int): Joint index in the body.

        Returns:
            float: Joint velocity.
        """
        return self.sim.get_joint_velocity(self.body_name, joint)

    def get_link_position(self, link: int) -> Tuple[float, float, float]:
        """Returns the cartesian position of a link as (x, y, z).

        Args:
            link (int): Link index in the body.

        Returns:
            (float, float, float): Link cartesian position.
        """
        return self.sim.get_link_position(self.body_name, link)

    def get_link_velocity(self, link: int) -> Tuple[float, float, float]:
        """Returns the velocity of a link as (vx, vy, vz).

        Args:
            link (int): Link index in the body.

        Returns:
            (float, float, float): Link velocity.
        """
        return self.sim.get_link_velocity(self.body_name, link)

    def control_joints(
        self,
        joint_indices: List[int],
        target_positions: List[float],
        target_forces: Optional[List[float]] = None,
    ) -> None:
        """Control the joints of the robot.

        Args:
            joint_indices (List[float]): List of joint indices.
            target_positions (List[float]): List of target positions/angles.
            target_forces (List[float], optional): List of target forces. Defaults to 150 per joint.
        """
        if target_forces is None:
            target_forces = [150] * len(joint_indices)

        self.sim.control_joints(
            body=self.body_name,
            joint_indices=joint_indices,
            target_positions=target_positions,
            target_forces=target_forces,
        )

    def get_num_joints(self) -> int:
        """Get the total number of joints.

        Returns:
            int: Total number of joints.
        """
        return self.sim.get_num_joints(self.body_name)

    def get_joint_name(self, joint: int) -> str:
        """Get the name of the joint.

        Args:
            joint (int): Joint index in the body.

        Returns:
            str: The name of the joint.
        """
        return self.sim.get_joint_name(self.body_name, joint)

    def get_joint_limits(self, joint: int) -> Tuple[float, float]:
        """Get lower and upper limits of the joint.

        Args:
            joint (int): Joint index in the body.

        Returns:
            (float, float): Lower and upper limit of the joint.
        """
        lower_limit = self.sim.get_joint_lower_limit(self.body_name, joint)
        upper_limit = self.sim.get_joint_upper_limit(self.body_name, joint)

        return lower_limit, upper_limit

    def set_joint_positions(
        self, joint_indices: List[int], positions: List[float]
    ) -> None:
        """(Re)Set joint positions.

        Args:
            joint_indices (List[int]): Joint indices in the body.
            positions (List[float]): Joint positions.
        """
        self.sim.set_joint_positions(
            body=self.body_name, joint_indices=joint_indices, positions=positions
        )


class RobotTaskEnv(gym.GoalEnv, ABC):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        self.seed()  # Required for init, can be changed later
        obs = self.reset()

        observation_shape = obs["observation"].shape
        achieved_goal_shape = obs["achieved_goal"].shape
        desired_goal_shape = obs["desired_goal"].shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-np.inf, np.inf, shape=observation_shape),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=achieved_goal_shape),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=desired_goal_shape),
            )
        )

        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        self.render = self.sim.render

    def _get_obs(self) -> dict:
        """Get observations.

        Returns:
            dict: Current observations.
                Observation keys: 'observation', 'achieved_goal', 'desired_goal'.
        """
        robot_obs = self.robot.get_obs()  # robot state
        task_obs = self.task.get_obs()  # object position, velocity, etc.
        observation = np.concatenate([robot_obs, task_obs])

        achieved_goal = self.task.get_achieved_goal()

        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal(),
        }

    def reset(self) -> dict:
        """Reset simulation.

        Returns:
            dict: Observations after reset.
                Observation keys: 'observation', 'achieved_goal', 'desired_goal'.
        """
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()

        return self._get_obs()

    def step(self, action) -> Tuple[dict, float, Union[float, bool], dict]:
        self.robot.set_action(action)
        self.sim.step()

        obs = self._get_obs()
        done = False
        info = {
            "is_success": self.task.is_success(
                obs["achieved_goal"], self.task.get_goal()
            ),
        }
        reward = self.task.compute_reward(
            obs["achieved_goal"], self.task.get_goal(), info
        )

        return obs, reward, done, info

    def seed(self, seed=None) -> int:
        """Setup the seed.

        Args:
            seed (int, optional): Seed.

        Returns:
            int: Seed.
        """
        return self.task.seed(seed)

    def close(self) -> None:
        """Close simulation."""
        self.sim.close()


class Task:
    """To be completed."""

    def get_goal(self):
        """Return the current goal."""
        raise NotImplementedError

    def get_obs(self):
        """Return the observation associated to the task."""
        raise NotImplementedError

    def get_achieved_goal(self):
        """Return the achieved goal."""
        raise NotImplementedError

    def reset(self):
        """Reset the task: sample a new goal."""
        pass

    def seed(self, seed):
        """Sets the seed for this env's random number."""
        self.np_random, seed = utils.seeding.np_random(seed)

    def is_success(self, achieved_goal, desired_goal):
        """Returns whether the acieved goal matches the desired goal or not."""
        raise NotImplementedError

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward associated to the achieved and the desired goal."""
        raise NotImplementedError
