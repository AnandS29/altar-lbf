import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5
    PUNISH_NORTH = 6
    PUNISH_SOUTH = 7
    PUNISH_WEST = 8
    PUNISH_EAST = 9

class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    AGENT = 2
    FOOD0 = 3
    FOOD1 = 4
    FOOD2 = 5


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None
        self.poisoned = 0
        self.marked = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0
        self.marked = False

    def increment_poisoned(self):
        if self.poisoned > 0:
            self.poisoned += 1
    
    def poison(self):
        self.poisoned += 1

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)
    
    def mark(self):
        self.marked = True

    def unmark(self):
        self.marked = False

    def is_marked(self):
        return self.marked

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class AltarForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD, Action.PUNISH_NORTH, Action.PUNISH_SOUTH, Action.PUNISH_WEST, Action.PUNISH_EAST]
    Observation = namedtuple(
        "Observation",
        ["field", "apples", "actions", "players", "game_over", "sight", "current_step", "altar"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self", "marked"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        max_player_level,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward=True,
        grid_observation=False,
        penalty=0.0,
        observe_altar=True,
        random_poison=True,
        observe_mark=True
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)
        self.apples = np.zeros(field_size, np.int32)
        self.apples_timer = np.zeros(field_size, np.int32)

        self.observe_altar = observe_altar
        self.mark = observe_mark
        self.random_poison = random_poison

        self.penalty = penalty
        
        self.max_food = max_food
        self.imposed_max_food_level = 1
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self.default_poison_apple = 2 # this apple is poisoned
        self.poison_apple = None
        self.poison_factor = 0.1 # how much the poison apple reduced the reward of eating an apple
        self.poison_threshold = 10 # how many steps till health effects begin to show

        self.punishment = 15
        self.cost_of_punishment = 5
        self.reward_for_correct_punishment = 15
        
        self.apple_expire_time = 7 # how many steps till an apple disappears

        self.altar = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation

        if self.mark:
            act_num = 10
        else:
            act_num = 6
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(act_num)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))

        self.viewer = None

        self.n_agents = len(self.players)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_food = self.max_food
            max_food_level = self.max_player_level * len(self.players)

            if self.observe_altar:
                if self.mark:
                    min_obs = [-1,-1,0] + [-1, -1, 0, 0] * max_food + [-1, -1, 0, 0] * len(self.players) # x,y,colour for altar + x,y,level, apple colour for each food + x,y,level,mark for each player
                    max_obs = [field_x-1, field_y-1, 2] + [field_x-1, field_y-1, max_food_level, 2] * max_food + [
                        field_x-1,
                        field_y-1,
                        self.max_player_level,
                        1
                    ] * len(self.players)
                else:
                    min_obs = [-1,-1,0] + [-1, -1, 0, 0] * max_food + [-1, -1, 0] * len(self.players) # x,y,colour for altar + x,y,level, apple colour for each food + x,y,level for each player
                    max_obs = [field_x-1, field_y-1, 2] + [field_x-1, field_y-1, max_food_level, 2] * max_food + [
                        field_x-1,
                        field_y-1,
                        self.max_player_level,
                    ] * len(self.players)
            else:
                if self.mark:
                    min_obs = [-1, -1, 0, 0] * max_food + [-1, -1, 0, 0] * len(self.players) # x,y,level, apple colour for each food + x,y,level for each player
                    max_obs = [field_x-1, field_y-1, max_food_level, 2] * max_food + [
                        field_x-1,
                        field_y-1,
                        self.max_player_level,
                        1
                    ] * len(self.players)
                else:
                    min_obs = [-1, -1, 0, 0] * max_food + [-1, -1, 0] * len(self.players) # x,y,level, apple colour for each food + x,y,level for each player
                    max_obs = [field_x-1, field_y-1, max_food_level, 2] * max_food + [
                        field_x-1,
                        field_y-1,
                        self.max_player_level,
                    ] * len(self.players)
        else: # TODO: implement grid observation with apple colour, altar, marking, etc.
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32) * self.max_player_level

            # foods layer: foods level
            max_food_level = self.max_player_level * len(self.players)
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min])
            max_obs = np.stack([agents_max, foods_max, access_max])

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

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
        env.apples = np.copy(obs.apples)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env.altar = obs.altar
        env.poison_apple = 2
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
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )
    
    def neighborhood_apples(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.apples[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.apples[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.apples[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

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
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self, max_food, max_level):
        food_count = 0
        attempts = 0
        min_level = max_level if self.force_coop else 1

        while food_count < max_food and attempts < 1000:
            attempts += 1
            row = self.np_random.randint(1, self.rows - 1)
            col = self.np_random.randint(1, self.cols - 1)

            # check if it has neighbors:
            # if (
            #     self.neighborhood(row, col).sum() > 0
            #     or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
            #     or not self._is_empty_location(row, col)
            # ):
            if (
                not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = (
                min_level
                if min_level == max_level
                # ! this is excluding food of level `max_level` but is kept for
                # ! consistency with prior LBF versions
                else self.np_random.randint(min_level, max_level)
            )
            self.apples[row,col] = np.random.randint(0, 3)
            self.apples_timer[row,col] = 0
            food_count += 1
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0 or (self.altar and self.altar[0] == row and self.altar[1] == col):
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True
    
    def spawn_altar(self):
        while True:
            row = self.np_random.randint(0, self.rows)
            col = self.np_random.randint(0, self.cols)
            if self._is_empty_location(row, col):
                self.altar = (row, col, self.poison_apple)
                break

    def spawn_players(self, max_player_level):
        for player in self.players:

            attempts = 0
            player.reward = 0
            player.poisoned = 0

            while attempts < 1000:
                row = self.np_random.randint(0, self.rows)
                col = self.np_random.randint(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.np_random.randint(1, max_player_level + 1),
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                # and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                # and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                # and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                # and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.field[player.position[0], player.position[1]] > 0
        elif action in [Action.PUNISH_EAST, Action.PUNISH_WEST, Action.PUNISH_NORTH, Action.PUNISH_SOUTH]:
            return self.mark # punishing is only valid if mark is enabled.


        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                    marked = int(a.is_marked()),
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            apples=np.copy(self.neighborhood_apples(*player.position, self.sight)), # TODO: double check
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
            altar=(self.altar)
        )

    def _make_gym_obs(self):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            if self.observe_altar:
                obs[0] = observation.altar[0]
                obs[1] = observation.altar[1]
                obs[2] = observation.altar[2]

                pre = 3
            else:
                pre = 0

            for i in range(self.max_food):
                obs[pre + 4 * i] = -1
                obs[pre + 4 * i + 1] = -1
                obs[pre + 4 * i + 2] = 0
                obs[pre + 4 * i + 3] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[pre + 4 * i] = y
                obs[pre + 4 * i + 1] = x
                obs[pre + 4 * i + 2] = observation.field[y, x]
                obs[pre + 4 * i + 3] = observation.apples[y, x]

            incr = 3
            if self.mark:
                incr = 4
            for i in range(len(self.players)):
                obs[pre + self.max_food * 4 + incr * i] = -1
                obs[pre + self.max_food * 4 + incr * i + 1] = -1
                obs[pre + self.max_food * 4 + incr * i + 2] = 0
                if self.mark:
                    obs[pre + self.max_food * 4 + incr * i + 3] = 0

            for i, p in enumerate(seen_players):
                obs[pre + self.max_food * 4 + incr * i] = p.position[0]
                obs[pre + self.max_food * 4 + incr * i + 1] = p.position[1]
                obs[pre + self.max_food * 4 + incr * i + 2] = p.level
                if self.mark:
                    obs[pre + self.max_food * 4 + incr * i + 3] = p.marked

            return obs

        def make_global_grid_arrays():
            """
            Create global arrays for grid observation space
            """
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)

            agents_layer = np.zeros(grid_shape, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                agents_layer[player_x + self.sight, player_y + self.sight] = player.level
            
            foods_layer = np.zeros(grid_shape, dtype=np.float32)
            foods_layer[self.sight:-self.sight, self.sight:-self.sight] = self.field.copy()

            access_layer = np.ones(grid_shape, dtype=np.float32)
            # out of bounds not accessible
            access_layer[:self.sight, :] = 0.0
            access_layer[-self.sight:, :] = 0.0
            access_layer[:, :self.sight] = 0.0
            access_layer[:, -self.sight:] = 0.0
            # agent locations are not accessible
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0
            # food locations are not accessible
            foods_x, foods_y = self.field.nonzero()
            for x, y in zip(foods_x, foods_y):
                access_layer[x + self.sight, y + self.sight] = 0.0
            
            return np.stack([agents_layer, foods_layer, access_layer])

        def get_agent_grid_bounds(agent_x, agent_y):
            return agent_x, agent_x + 2 * self.sight + 1, agent_y, agent_y + 2 * self.sight + 1
        
        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        observations = [self._make_obs(player) for player in self.players]
        if self._grid_observation: # TODO: implement for apples
            layers = make_global_grid_arrays()
            agents_bounds = [get_agent_grid_bounds(*player.position) for player in self.players]
            nobs = tuple([layers[:, start_x:end_x, start_y:end_y] for start_x, end_x, start_y, end_y in agents_bounds])
        else:
            nobs = tuple([make_obs_array(obs) for obs in observations]) 
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {}
        
        # check the space of obs
        for i, obs in  enumerate(nobs):
            assert self.observation_space[i].contains(obs), \
                f"obs space error: obs: {obs}, obs_space: {self.observation_space[i]}"
        
        return nobs, nreward, ndone, ninfo
    
    def set_poison_apple(self):
        if self.random_poison:
            self.poison_apple = np.random.randint(0, 3)
        else:
            self.poison_apple = self.default_poison_apple

    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.apples = np.zeros(self.field_size, np.int32)
        self.apples_timer = np.zeros(self.field_size, np.int32)
        self.set_poison_apple()
        self.spawn_altar()
        self.spawn_players(self.max_player_level)
        player_levels = sorted([player.level for player in self.players])
        self.spawn_food(
            self.max_food, max_level=(self.imposed_max_food_level if self.imposed_max_food_level is not None else sum(player_levels[:3]))
        )
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()


        nobs, _, _, _ = self._make_gym_obs()
        return nobs
    
    def find_players_at_loc(self, players, loc):
        return [player for player in players if player.position == loc]

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()
        punished_players = [] # Can be punished by multiple people

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
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
            elif action in [Action.PUNISH_EAST, Action.PUNISH_WEST, Action.PUNISH_NORTH, Action.PUNISH_SOUTH]:
                collisions[player.position].append(player)
                # Get punishing location
                if action == Action.PUNISH_EAST:
                    punishing_location = (player.position[0], player.position[1] + 1)
                elif action == Action.PUNISH_WEST:
                    punishing_location = (player.position[0], player.position[1] - 1)
                elif action == Action.PUNISH_NORTH:
                    punishing_location = (player.position[0] - 1, player.position[1])
                elif action == Action.PUNISH_SOUTH: 
                    punishing_location = (player.position[0] + 1, player.position[1])

                # Get players at punishing location
                punished_players_at_loc = self.find_players_at_loc(self.players, punishing_location)
                player.reward -= self.cost_of_punishment

                for p in punished_players_at_loc:
                    punished_players.append(p)
                    if p.is_marked():
                        player.reward += self.reward_for_correct_punishment

        # punish players
        for player in punished_players:
            player.reward -= self.punishment
            player.unmark()
                
        # and do movements for non colliding players

        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than one player will arrive at location
                continue
            v[0].position = k


        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = player.position[0], player.position[1]
            food = self.field[frow, fcol]
            is_food_poisoned = self.apples[frow, fcol] == self.poison_apple

            # adj_players = self.adjacent_players(frow, fcol)
            # adj_players = [
            #     p for p in adj_players if p in loading_players or p is player
            # ]

            # adj_player_level = sum([a.level for a in adj_players])

            # loading_players = loading_players - set(adj_players)

            # if adj_player_level < food:
            #     # failed to load
            #     for a in adj_players:
            #         a.reward -= self.penalty
            #     continue

            # else the food was loaded and each player scores points
            # for a in adj_players:
            #     r = float(a.level * food)
            #     if a.poisoned > self.poison_threshold:
            #         r *= self.poison_factor
            #     a.reward = r
            #     if self._normalize_reward:
            #         a.reward = a.reward / float(
            #             adj_player_level * self._food_spawned
            #         )  # normalize reward
            #     if is_food_poisoned:
            #         a.poison()
            # and the food is removed
            r = 1
            if player.poisoned > self.poison_threshold:
                r *= self.poison_factor
            player.reward += r
            # if self._normalize_reward:
            #     player.reward = player.reward / float(
            #         adj_player_level * self._food_spawned
            #     )  # normalize reward
            if is_food_poisoned:
                player.poison()
                if self.mark:
                    player.mark()
            self.remove_apple(frow, fcol)

        self._game_over = (
            # self.field.sum() == 0 or self._max_episode_steps <= self.current_step
            self._max_episode_steps <= self.current_step
        )

        # Increment apple timers
        self.increment_timer()

        # Expire apples
        self.expire_apples()

        # spawn new food if needed
        if self.max_food - self.field.sum() > 0:
            player_levels = sorted([player.level for player in self.players])
            self.spawn_food(
                self.max_food - self.field.sum(), max_level=(self.imposed_max_food_level if self.imposed_max_food_level is not None else sum(player_levels[:3]))
            )

        self._gen_valid_moves()

        for p in self.players:
            p.score += p.reward
            # Update poison counter
            p.increment_poisoned() # only increment if poisoned > 0 (ie they've been poisoned)

        return self._make_gym_obs()
    
    def increment_timer(self):
            self.apples_timer[self.field > 0] += 1

    def remove_apple(self,row,col):
        self.apples[row,col] = 0
        self.apples_timer[row,col] = 0
        self.field[row,col] = 0

    def expire_apples(self):
        # expire apples
        expired_apples = np.where(self.apples_timer >= self.apple_expire_time)
        for row,col in zip(expired_apples[0],expired_apples[1]):
            self.remove_apple(row,col)

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
