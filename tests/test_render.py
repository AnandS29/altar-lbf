import gym
import numpy as np
import lbforaging
import time

# arg parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--both", action="store_true", help="Run both agents")
parser.add_argument("--render", action="store_true", help="Render the environment")

args = parser.parse_args()
# Test rednering

env = gym.make('AltarForaging-7x7-2p-4f-altar-rand-mark-v2')
obs = env.reset()

def nice_print_ind(obs):
    altar_x, altar_y, altar_colour = obs[0], obs[1], obs[2]
    berries = []
    for i in range(4):
        berries.append((obs[3+3*i], obs[4+3*i], obs[5+3*i], obs[6+3*i]))
    pre = 4*4 + 3
    players = []
    for i in range(2):
        players.append((obs[pre+4*i], obs[pre+4*i+1], obs[pre+4*i+2], obs[pre+4*i+3]))

    print("Altar: ({}, {}) {}".format(altar_x, altar_y, altar_colour))
    print()
    for b in berries:
        print("Berry: ({}, {}) lvl={}, color={}".format(b[0], b[1], b[2], b[3]))
    print()
    i = 0
    for p in players:
        if i == 0:
            print("Player (ME): ({}, {}) lvl={}, marked={}".format(p[0], p[1], p[2], p[3]))
        else:
            print("Player: ({}, {}) lvl={}, marked={}".format(p[0], p[1], p[2], p[3]))
        i += 1

def nice_print(obs):
    print("Agent 1")
    print(nice_print_ind(obs[0]))
    print(obs[0])
    print()
    print("Agent 2")
    print(nice_print_ind(obs[1]))
    print(obs[1])
    print()

if args.render:
    env.render()
else:
    nice_print(obs)

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
    if args.both:
        action2 = input("Action 2: ")
        action2 = code_action(action2)
    else:
        action2 = np.random.randint(0, 6)
    action = (action1, action2)
    obs, reward, done, info = env.step(action)
    cum_reward[0] += reward[0]
    cum_reward[1] += reward[1]
    print(cum_reward)

    if args.render:
        env.render()
    else:
        nice_print(obs)

# sleep for 10 seconds
time.sleep(10)