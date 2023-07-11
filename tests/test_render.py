import gym
import numpy as np
import lbforaging
import time

# Test rednering

env = gym.make('AltarForaging-7x7-2p-4f-altar-rand-mark-v2')
obs = env.reset()

env.render()

def code_action(action):
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
    elif action == "up":
        action = 6
    elif action == "right":
        action = 9
    elif action == "down":
        action = 7
    elif action == "left":
        action = 8
    else:
        action = 0
    return action

cum_reward = [0,0]
# Sample random actions
for _ in range(1000):
    action1 = input("Action 1: ")
    action1 = code_action(action1)
    # action2 = input("Action 2: ")
    # action2 = code_action(action2)

    action2 = np.random.randint(0, 6)
    action = (action1, action2)
    obs, reward, done, info = env.step(action)
    cum_reward[0] += reward[0]
    cum_reward[1] += reward[1]
    print(cum_reward)

    env.render()

# sleep for 10 seconds
time.sleep(10)