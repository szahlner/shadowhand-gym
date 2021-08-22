import gym
import shadowhand_gym


env = gym.make("ShadowHandBlock-v1", render=True)

obs = env.reset()
done = False
while not done:
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

env.close()
