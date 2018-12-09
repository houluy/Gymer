import gym
from pprint import pprint
env = gym.make('CartPole-v0')

observation = env.reset()
for step in range(10):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

