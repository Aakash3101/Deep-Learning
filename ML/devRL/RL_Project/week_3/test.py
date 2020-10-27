import gym
import numpy as np
from qlearning import QLearningAgent
import matplotlib.pyplot as plt


env = gym.make('LunarLander-v2')
env.reset()
n_actions = env.action_space.n
states = []
rewards = 0

for _ in range(2000):
    env.render()
    a = 0
    s, r, done, _ = env.step(a)
    states.append(s)
    rewards += r

env.close()

print(states[-1])
print(rewards)

