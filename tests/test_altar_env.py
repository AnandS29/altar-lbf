import gym
import numpy as np
import lbforaging
import time

env = gym.make('AltarForaging-10x10-2p-10f-altar-rand-v2')
obs = env.reset()
print('Starting test')

def shortest_path(curr_x,curr_y,goal_x,goal_y, obstacles, size):
    # Returns the shortest path from (curr_x,curr_y) to (goal_x,goal_y)
    # Avoids obstacles
    # Returns a list of (x,y) tuples
    # Uses BFS

    # Queue of (x,y) tuples
    queue = [(curr_x,curr_y)]
    visited = set()
    visited.add((curr_x,curr_y))
    parents = {}
    parents[(curr_x,curr_y)] = None
    found = False
    while len(queue) > 0:
        curr = queue.pop(0)
        if curr[0] == goal_x and curr[1] == goal_y:
            found

def simple_policy(obss):
    obs = obss[0] # food and player info is the same for both players
    altar_x, altar_y, altar_colour = obs[0:3]
    foods = []
    players = []
    for i in range(10):
        foods.append(obs[3+4*i:3+4*(i+1)])

    for i in range(2):
        players.append(obs[43+3*i:43+3*(i+1)])

    # Find closest food
    closest = {}
    for player in players:
        player_x, player_y, player_colour = player[0:3]
        closest_food = foods[0]
        closest_food_dist = np.inf
        for food in foods:
            dist = np.linalg.norm(np.array(food[0:2]) - np.array([player_x, player_y]))
            food_colour = food[2]
            if dist < closest_food_dist and food_colour != altar_colour:
                closest_food = food
                closest_food_dist = dist
        closest[tuple(player)] = closest_food
    
    # Move towards closest food while avoiding other player
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5

    def is_player_there(x,y):
        for player in players:
            player_x, player_y, player_colour = player[0:3]
            if player_x == x and player_y == y:
                return True
        return False
    
    def next_loc(x,y,action):
        if action == NORTH:
            return x, y+1
        elif action == SOUTH:
            return x, y-1
        elif action == EAST:
            return x+1, y
        elif action == WEST:
            return x-1, y
        else:
            return x, y

    actions = []
    for player in players:
        action = None
        food = closest[tuple(player)]
        player_x, player_y, player_colour = player[0:3]
        next_to_food = False
        # check if food is above or below or left or right
        if player_x == food[0]:
            if player_y == food[1] + 1:
                next_to_food = True
            elif player_y == food[1] - 1:
                next_to_food = True
        elif player_y == food[1]:
            if player_x == food[0] + 1:
                next_to_food = True
            elif player_x == food[0] - 1:
                next_to_food = True

        if food[2] == altar_colour:
            action = NONE
        elif not next_to_food:
            # Order directions by distance to food
            directions = [NORTH, SOUTH, EAST, WEST]
            directions.sort(key=lambda x: np.linalg.norm(np.array(next_loc(player_x,player_y,x)) - np.array(food[0:2])))
            action = None
            for direction in directions:
                next_x, next_y = next_loc(player_x,player_y,direction)
                if not is_player_there(next_x,next_y):
                    action = direction
                    break
        else:
            action = LOAD
        actions.append(action)
    return tuple(actions)
    
ep_rews = []
n = 100
for i in range(n):
    # Sample random actions
    obss = env.reset()
    cum_rewards = [0 for _ in range(len(obss))]
    for _ in range(1000):
        action = simple_policy(obss)
        obss, rewards, dones, infos = env.step(action)
        for i in range(len(obss)):
            cum_rewards[i] += rewards[i]

    avg_reward = np.average(cum_rewards)
    ep_rews.append(avg_reward)

avg_eps_reward = np.average(ep_rews)

print("Test finished")
print("Average reward over {} episodes: {}".format(n, avg_eps_reward))