from abc import ABC

from shadowhand_gym.envs.core import RobotTaskEnv
from shadowhand_gym.pybullet import PyBullet
from shadowhand_gym.envs.robots import ShadowHand
from shadowhand_gym.envs.tasks import Block


class ShadowHandBlockEnv(RobotTaskEnv, ABC):
    def __init__(self, render: bool = False, reward_type: str = "sparse") -> None:
        """Block manipulation task with Shadow Dexterous Hand robot.

        Args:
            render (bool, optional): Activate rendering. Defaults to False.
            reward_type (str, optional): 'sparse' or 'dense'. Defaults to 'sparse'.
        """
        self.sim = PyBullet(render=render)
        self.robot = ShadowHand(
            sim=self.sim,
            base_position=[0.0, 0.0, 0.0],
            base_orientation=[0.5, -0.5, 0.5, -0.5],
        )
        self.task = Block(sim=self.sim, robot=self.robot, reward_type=reward_type)
        RobotTaskEnv.__init__(self)
