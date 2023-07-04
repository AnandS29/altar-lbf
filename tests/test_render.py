import gym
import numpy as np
import lbforaging
import time

# Test rednering

env = gym.make('AltarForaging-10x10-2p-10f-altar-rand-v2')
obs = env.reset()
print(obs)
env.render()

# Sample random actions
for _ in range(1000):
    action = env.action_space.sample()
    time.sleep(0.1)
    env.step(action)
    env.render()

# sleep for 10 seconds
time.sleep(10)