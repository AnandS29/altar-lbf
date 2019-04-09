import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from random import randint

import numpy as np


class Action(Enum):
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.history = None
        self.current_step = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv:
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple("Observation", ['field', 'actions', 'players', 'game_over', 'sight', 'current_step'])
    PlayerObservation = namedtuple("PlayerObservation", ['position', 'level', 'history', 'score',
                                                         'is_self'])  # score is available only if is_self

    def __init__(self, players, max_player_level, field_size, max_food, sight):
        self.logger = logging.getLogger(__name__)
        self.players = players

        if field_size:
            self.field = np.zeros(field_size, np.int32)

        self.max_food = max_food
        self.max_player_level = max_player_level
        self.sight = sight
        self._game_over = None

        self._rendering_initialized = False

        self._valid_actions = None

    @classmethod
    def from_obs(cls, obs):

        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {player: [action for action in Action if self._is_valid_action(player, action)] for player
                               in self.players}

    def neighborhood(self, row, col, distance=1):

        return self.field[
               max(row - distance, 0):min(row + distance + 1, self.rows),
               max(col - distance, 0):min(col + distance + 1, self.cols)
               ]

    def adjacent_food(self, row, col):
        return self.field[max(row - 1, 0), col] + self.field[min(row + 1, self.rows - 1), col] + self.field[
            row, max(col - 1, 0)] + self.field[row, min(col + 1, self.cols - 1)]

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [player for player in self.players if
                abs(player.position[0] - row) == 1 and player.position[1] == col or
                abs(player.position[1] - col) == 1 and player.position[0] == row
                ]

    def spawn_food(self, max_food, max_level):

        food_count = 0
        attempts = 0

        while food_count < max_food and attempts < 1000:
            attempts += 1
            row = randint(1, self.rows - 2)
            col = randint(1, self.cols - 2)

            # check if it has neighbors:
            if self.neighborhood(row, col, distance=2).sum() > 0 or not self._is_empty_location(row, col):
                continue

            self.field[row, col] = randint(1, max_level)
            food_count += 1

    def _is_empty_location(self, row, col):

        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, max_player_level):

        for player in self.players:

            attempts = 0

            while attempts < 1000:
                row = randint(0, self.rows - 1)
                col = randint(0, self.cols - 1)
                if self._is_empty_location(row, col):
                    player.setup((row, col), randint(1, max_player_level), self.field_size)
                    break
                attempts += 1

    def _is_valid_action(self, player, action):

        if action == Action.NORTH:
            return player.position[0] > 0 and self.field[player.position[0] - 1, player.position[1]] == 0
        elif action == Action.SOUTH:
            return player.position[0] < self.rows - 1 and self.field[player.position[0] + 1, player.position[1]] == 0
        elif action == Action.WEST:
            return player.position[1] > 0 and self.field[player.position[0], player.position[1] - 1] == 0
        elif action == Action.EAST:
            return player.position[1] < self.cols - 1 and self.field[player.position[0], player.position[1] + 1] == 0
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return position[0] - center[0] + min(sight, center[0]), position[1] - center[1] + min(sight, center[1])

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[self.PlayerObservation(
                position=self._transform_to_neighborhood(player.position, self.sight, a.position), level=a.level,
                is_self=a == player, history=a.history, score=a.score if a == player else None) for a in self.players if
                min(self._transform_to_neighborhood(player.position, self.sight, a.position)) >= 0],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.max_player_level)
        self.spawn_food(self.max_food, max_level=sum([player.level for player in self.players]))
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        return [self._make_obs(player) for player in self.players]

    def step(self, actions):
        self.current_step += 1

        # check if actions are valid
        for player, action in zip(self.players, actions):
            if action not in self._valid_actions[player]:
                self.logger.error('{}{} attempted invalid action {}.'.format(player.name, player.position, action))
                self.logger.error(self.field)
                raise ValueError("Invalid action attempted")

            # also give a negative reward if action is not LOAD
            if action != Action.LOAD:
                player.score -= 0.01

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            # if action == Action.NONE:
            # 	collisions[player.position].append(player)
            if action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [p for p in adj_players if p in loading_players or p is player]

            adj_player_level = sum([a.level for a in adj_players])

            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.score += food
            # and the food is removed
            self.field[frow, fcol] = 0

        self._game_over = self.field.sum() == 0
        self._gen_valid_moves()

        return [self._make_obs(player) for player in self.players]

    def _init_render(self):
        from .rendering import Viewer
        self.viewer = Viewer((self.rows, self.cols))

    def render(self):
        if not self._rendering_initialized:
            self._init_render()

        self.viewer.render(self)