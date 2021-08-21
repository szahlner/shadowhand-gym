from abc import ABC

from shadowhand_gym.envs.core import RobotTaskEnv
from shadowhand_gym.pybullet import PyBullet
from shadowhand_gym.envs.robots import ShadowHand
from shadowhand_gym.envs.tasks import Reach


class ShadowHandReachEnv(RobotTaskEnv, ABC):
    def __init__(
        self,
        render: bool = False,
        reward_type: str = "sparse",
        difficult_mode: str = "easy",
    ) -> None:
        """Reach task with Shadow Dexterous Hand robot.

        Args:
            render (bool, optional): Activate rendering. Defaults to False.
            reward_type (str, optional): Choose from 'sparse' or 'dense'. Defaults to 'sparse'.
            difficult_mode (str, optional): Choose from 'easy' or 'hard'. Defaults to 'easy'.
                'easy': only choose between one of the 4 given fingertip positions.
                'hard': choose between all of the 4 given fingertip positions (per finger).
        """
        self.sim = PyBullet(render=render)
        self.robot = ShadowHand(
            sim=self.sim,
            base_position=[0.0, 0.0, 0.0],
            base_orientation=[0.5, -0.5, 0.5, -0.5],
        )
        self.task = Reach(
            sim=self.sim,
            robot=self.robot,
            reward_type=reward_type,
            difficult_mode=difficult_mode,
        )
        RobotTaskEnv.__init__(self)
