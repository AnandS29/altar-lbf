import gym
import numpy as np
import lbforaging
import time

# Test rednering

env = gym.make('AltarForaging-7x7-2p-4f-altar-rand-v2')
obs = env.reset()

env.render()

# Sample random actions
for _ in range(1000):
    action = input("Action: ")
    if action == "":
        action = 0
    elif action == "w":
        action = 1
    elif action == "d":
        action = 4
    elif action == "s":
        action = 2
    elif action == "a":
        action = 3
    elif action == "l":
        action = 5
    else:
        action = 0
    
    action = (action, 0)
    print(action)
    obs, reward, done, info = env.step(action)
    print(reward)

    obs1, obs2 = obs
    print(obs1)
    print(obs2)
    env.render()

# sleep for 10 seconds
time.sleep(10)