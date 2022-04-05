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
            [0.4151469, -0.00737783, 0.04983524],  # Index Finger
            [0.41973359, 0.0160899, 0.0443932],  # Middle Finger
            [0.40823004, 0.03960472, 0.04899406],  # Ring Finger
            [0.3939393, 0.06979136, 0.04983475],  # Little Finger
            [0.37106937, -0.0163987, 0.04818256],  # Thumb
        ],
        # Goal 2 (pistol)
        [
            [0.43758884, -0.00128918, 0.02065529],
            [0.43820698, 0.02591066, 0.01888607],
            [0.32209741, 0.02534194, 0.04953822],
            [0.3107758, 0.04531284, 0.05274596],
            [0.37454418, -0.03739185, 0.03882779],
        ],
        # Goal 3 (hanging)
        [
            [0.370413, -0.01039512, 0.08245648],
            [0.36341932, 0.00963706, 0.07992665],
            [0.35482527, 0.03092836, 0.07912724],
            [0.35291384, 0.05460698, 0.07967867],
            [0.37061437, -0.02584797, 0.04864619],
        ],
        # Goal 4 (rock)
        [
            [0.42523561, -0.00581391, 0.05646441],
            [0.33373276, 0.00393033, 0.04919242],
            [0.3372073, 0.0257887, 0.049753],
            [0.40934382, 0.06401634, 0.06163007],
            [0.36643599, -0.03822211, 0.05919186],
        ],
    ]

    def __init__(
        self,
        sim: PyBullet,
        robot: ShadowHand,
        reward_type: str = "sparse",
        distance_threshold: float = 0.03,
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
