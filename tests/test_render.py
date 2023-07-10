import gym
import numpy as np
import lbforaging
import time

# Test rednering

env = gym.make('AltarForaging-7x7-2p-4f-rand-v2')
obs = env.reset()

env.render()

cum_reward = [0,0]
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
    
    action = (action, np.random.randint(0, 6))

    print(action)
    obs, reward, done, info = env.step(action)
    cum_reward[0] += reward[0]
    cum_reward[1] += reward[1]
    print(cum_reward)

    env.render()

# sleep for 10 seconds
time.sleep(10)