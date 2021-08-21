import numpy as np

from shadowhand_gym.envs.core import Task
from shadowhand_gym.pybullet import PyBullet
from shadowhand_gym.envs.robots.shadowhand import ShadowHand


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance to goal.

    Args:
        a (np.ndarray): Achieved goal (fingertip positions).
        b (np.ndarray): Desired goal (fingertip positions).

    Returns:
        float: Distance to goal.
    """
    assert a.shape == b.shape, "Shape of 'a' must match shape of 'b'"
    return np.linalg.norm(a - b, axis=-1)


class Reach(Task):

    GOAL_POSITIONS = [
        # Goal 1 (normal)
        [
            [0.415, -0.007, 0.050],  # Index Finger
            [0.420, 0.016, 0.044],  # Middle Finger
            [0.408, 0.040, 0.050],  # Ring Finger
            [0.394, 0.070, 0.050],  # Little Finger
            [0.371, -0.016, 0.048],  # Thumb
        ],
        # Goal 2 (pistol)
        [
            [0.438, -0.001, 0.021],
            [0.438, 0.026, 0.019],
            [0.322, 0.025, 0.050],
            [0.311, 0.045, 0.053],
            [0.375, -0.037, 0.039],
        ],
        # Goal 3 (hanging)
        [
            [0.37, -0.01, 0.082],
            [0.363, 0.01, 0.080],
            [0.355, 0.031, 0.079],
            [0.353, 0.055, 0.080],
            [0.371, -0.026, 0.049],
        ],
        # Goal 4 (rock)
        [
            [0.425, -0.006, 0.056],
            [0.324, -0.004, 0.049],
            [0.317, 0.018, 0.050],
            [0.409, 0.064, 0.062],
            [0.366, -0.038, 0.060],
        ],
    ]

    def __init__(
        self,
        sim: PyBullet,
        robot: ShadowHand,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        difficult_mode: str = "easy",
    ) -> None:
        """Shadow dexterous hand reach task.

        Args:
            sim (PyBullet): PyBullet client to interact with the simulator.
            robot (ShadowHand): Shadow dexterous hand robot.
            reward_type (str, optional): Reward type. Choose from 'dense' or 'sparse'. Defaults to 'sparse'.
            distance_threshold (float, optional): Distance threshold to determine between success and failure.
            difficult_mode (str, optional): Difficulty. Choose from 'easy' or 'hard'. Defaults to 'easy'.
                'easy': only choose between one of the 4 given fingertip positions.
                'hard': choose between all of the 4 given fingertip positions (per finger).
        """
        assert reward_type in [
            "dense",
            "sparse",
        ], "Reward type must be 'dense' or 'sparse'"
        assert difficult_mode in [
            "easy",
            "hard",
        ], "Difficult mode must be in 'easy' or 'hard'"

        self.sim = sim
        self.robot = robot
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.difficult_mode = difficult_mode

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(
                target_position=[0.25, 0.0, 0.0], distance=0.5, yaw=45, pitch=-40
            )

        self.goal = None

    def _create_scene(self) -> None:
        """Create scene.

        Add (ghost) objects and stuff that is not included in the robot URDF file."""
        self.sim.create_sphere(
            body_name="target_index_finger",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=[0.0, 0.0, 0.0],
            rgba_color=[0.1, 0.1, 0.9, 0.5],
        )
        self.sim.create_sphere(
            body_name="target_middle_finger",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=[0.0, 0.0, 0.0],
            rgba_color=[0.1, 0.9, 0.1, 0.5],
        )
        self.sim.create_sphere(
            body_name="target_ring_finger",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=[0.0, 0.0, 0.0],
            rgba_color=[0.9, 0.1, 0.1, 0.5],
        )
        self.sim.create_sphere(
            body_name="target_little_finger",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=[0.0, 0.0, 0.0],
            rgba_color=[0.1, 0.9, 0.9, 0.5],
        )
        self.sim.create_sphere(
            body_name="target_thumb",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=[0.0, 0.0, 0.0],
            rgba_color=[0.9, 0.1, 0.9, 0.5],
        )

    def get_goal(self) -> np.ndarray:
        """Return goal.

        Returns:
            np.ndarray: Cartesian positions of the fingertips (target).
        """
        return self.goal.copy()

    def get_obs(self) -> np.ndarray:
        """Return task specific observations.

        Returns:
            np.ndarray: Empty.
        """
        return np.array([])  # no task specific observations

    def get_achieved_goal(self) -> np.ndarray:
        """Return achieved goal.

        Returns:
            np.ndarray: Cartesian positions of the fingertips.
        """
        positions = self.robot.get_fingertip_positions()
        return positions.flatten()

    def reset(self) -> None:
        """Reset task."""
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target_index_finger", self.goal[0], [0, 0, 0, 1])
        self.sim.set_base_pose("target_middle_finger", self.goal[1], [0, 0, 0, 1])
        self.sim.set_base_pose("target_ring_finger", self.goal[2], [0, 0, 0, 1])
        self.sim.set_base_pose("target_little_finger", self.goal[3], [0, 0, 0, 1])
        self.sim.set_base_pose("target_thumb", self.goal[4], [0, 0, 0, 1])
        self.goal = self.goal.flatten()

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal.

        Returns:
            np.ndarray: Cartesian positions of the fingertips (target).
        """
        # Difficult mode: easy
        # Only choose between one of the 4 given fingertip positions
        goal_choice = self.np_random.choice(
            [n for n in range(len(self.GOAL_POSITIONS))]
        )
        goal = np.array(self.GOAL_POSITIONS[goal_choice])

        if self.difficult_mode == "hard":
            # Difficult mode: hard
            # Choose between all of the 4 given fingertip positions (per finger)
            for n in range(5):
                goal_choice = self.np_random.choice(
                    [k for k in range(len(self.GOAL_POSITIONS))]
                )
                goal[n] = np.array(self.GOAL_POSITIONS[goal_choice][n])

        if self.np_random.uniform() < 0.1:
            goal = self.robot.get_fingertip_positions()

        return goal.copy()

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
