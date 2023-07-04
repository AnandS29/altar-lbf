import gym
import numpy as np
import lbforaging
import time

# Test rednering

env = gym.make('AltarForaging-8x8-2p-3f-v2')
obs = env.reset()
print(obs)
env.render()

# Sample random actions
for _ in range(100):
    action = env.action_space.sample()
    time.sleep(0.1)
    env.step(action)
    env.render()

# sleep for 10 seconds
time.sleep(10)