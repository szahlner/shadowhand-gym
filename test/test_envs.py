import gym
import shadowhand_gym

from typing import Union


def run_env(env: Union[gym.Env, gym.GoalEnv]):
    """Tests running shadowhand-gym environments.

    Args:
        env (Union[Env, GoalEnv]): Environment to test.
    """
    done = False
    env.reset()

    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)

    env.close()


def test_reach():
    env = gym.make("ShadowHandReach-v1")
    run_env(env)


def test_block():
    env = gym.make("ShadowHandBlock-v1")
    run_env(env)
