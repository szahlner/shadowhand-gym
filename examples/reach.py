import gym
import shadowhand_gym


# Also try to change the difficulty to hard: difficult_mode="hard"
env = gym.make("ShadowHandReach-v1", render=True, difficult_mode="easy")

obs = env.reset()
done = False
while not done:
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

env.close()
