import logging

from briscola_rl.players.base_player import BasePlayer
from gymnasium import spaces

from briscola_rl.players.epsgreedy_player import EpsGreedyPlayer
from briscola_rl.state import PublicState

from briscola_rl.game_rules import select_winner
from briscola_rl.cards import *
import gymnasium as gym

from briscola_rl.players.random_player import PseudoRandomPlayer
from briscola_rl.players.human_player import HumanPlayer


class BriscolaCustomEnemyPlayer(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, other_player: BasePlayer, played: bool = True):
        # Define the action space
        self.action_space = spaces.Discrete(3)  # drop i-th card
        # Define the observation space
        # Card space: (value, seed, points)
        card_space = spaces.MultiDiscrete([14, 5, 12])
        self.observation_space_nested = spaces.Dict({
            'my_points': spaces.Discrete(121),
            'other_points': spaces.Discrete(121),
            'hand_size': spaces.Discrete(4),
            'other_hand_size': spaces.Discrete(4),
            'remaining_deck_cards': spaces.Discrete(41),
            'hand': spaces.Tuple([card_space, card_space, card_space]),
            'table': spaces.Tuple([card_space, card_space]),
            'turn': spaces.Discrete(40),
            'briscola': card_space,
            'order': spaces.Discrete(2)
        } | ({
            'my_played': spaces.Tuple([card_space] * 40),
            'other_played': spaces.Tuple([card_space] * 40),
        } if played else {}))
        self.observation_space = spaces.flatten_space(self.observation_space_nested)
        self.played = played

        self.my_player: BasePlayer = HumanPlayer()
        self.other_player = other_player
        self.players = [self.my_player, self.other_player]
        self.reward_range = (-22, 22)
        self.deck = None
        self.briscola: Card = None
        self.__logger = logging.getLogger('Briscola')
        self.turn_my_player = 0  # I start

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._turn = 0
        self.my_player.reset_player()
        self.other_player.reset_player()
        self.deck = Deck(seed=seed)
        self._my_played = []
        self._other_played = []
        self._table = []
        self._my_points = 0
        self._other_points = 0
        self._points = [0, 0]
        self._turn_my_player = self.np_random.integers(0, high=1, endpoint=True)
        self._log_turn_leader()
        self.players = [self.my_player, self.other_player]
        self.briscola: Card = self.deck.draw()
        self.__logger.info(f'Briscola is {self.briscola}')
        for _ in range(3):
            self.my_player.hand.append(self.deck.draw())
        for _ in range(3):
            self.other_player.hand.append(self.deck.draw())
        self._log_hands()
        self.deck.cards.append(self.briscola)
        if self._turn_my_player == 1:
            other_card = self.other_player.play_card(self.public_state())
            self._table.append(other_card)

        return self._get_obs(), self._get_info()

    def _log_hands(self):
        self.__logger.info(f'Player hand is: {self.my_player.hand}')
        self.__logger.info(f'Enemy hand is: {self.other_player.hand}')

    def _log_turn_leader(self):
        self.__logger.info(f'Starts {self.players[self._turn_my_player].name}')

    def _get_obs(self):
        return spaces.flatten(self.observation_space_nested, self.public_state().as_dict(played=self.played))

    def _get_info(self):
        return dict()

    def step(self, action):
        assert action in self.action_space
        self._turn += 1
        while action >= len(self.my_player.hand):
            action = action - 1
        my_card = self.my_player.hand.pop(action)
        self.__logger.info(f"Agent plays {my_card}")
        self._table.append(my_card)
        if self._turn_my_player == 0:
            other_card = self.other_player.play_card(self.public_state())
            self._table.append(other_card)
        self.__logger.info(f'Table: {self._table}')
        # i_winner: 0 -> Wins first player, 1 -> Wins second player
        i_winner = select_winner(self._table, self.briscola)
        reward = self._state_update_after_winner(i_winner)
        self._draw_phase()
        self._log_turn_leader()
        if self._turn_my_player == 1 and not len(self.other_player.hand) == 0:
            other_card = self.other_player.play_card(self.public_state())
            self._table.append(other_card)
        return self._get_obs(), reward, self.is_terminated(), False, self._get_info()

    def _state_update_after_winner(self, i_winner: int):
        """
        :param i_winner: 0 -> Wins first, 1 -> Wins second
        :return: The reward
        """
        i_winner = 0 if self._turn_my_player == i_winner else 1
        self.__logger.info(f'Turn Winner is {self.players[i_winner].name}')
        reward = gained_points = sum(values_points[c.value] for c in self._table)
        self._points[i_winner] += gained_points
        self._my_played.append(self._table[self._turn_my_player])
        self._other_played.append(self._table[1 - self._turn_my_player])
        gained_points_my_player = gained_points_other_player = gained_points
        if i_winner == 0:
            self._my_points += gained_points
            self._turn_my_player = 0
            gained_points_other_player = gained_points_other_player * -1
        else:
            self._other_points += gained_points
            self._turn_my_player = 1
            gained_points_my_player = gained_points_my_player * -1
            reward = reward * -1
        self.my_player.notify_turn_winner(gained_points_my_player)
        self.other_player.notify_turn_winner(gained_points_other_player)
        self._table = []
        self.__logger.info(f'Winner gained {gained_points} points')
        self.__logger.info(f'Current game points: {self._points}')
        return reward

    def _draw_phase(self):
        if not self.deck.is_empty():
            c1 = self.deck.draw()
            c2 = self.deck.draw()
            if self._turn_my_player == 0:
                c_my_player = c1
                c_other_player = c2
            else:
                c_other_player = c1
                c_my_player = c2
            self.my_player.hand.append(c_my_player)
            self.other_player.hand.append(c_other_player)
            self._log_hands()

    def public_state(self):
        return PublicState(self._my_points, self._other_points, self.my_player.hand,
                           len(self.other_player.hand), len(self.deck.cards),
                           self._table, self._my_played, self._other_played,
                           self._turn, self.briscola, self._turn_my_player)

    def is_terminated(self):
        return any(p > 60 for p in self._points) or \
            (self.deck.is_empty() and all(len(p.hand) == 0 for p in self.players))

    def render(self, mode="human"):
        pass


class BriscolaRandomPlayer(BriscolaCustomEnemyPlayer):

    def __init__(self, played: bool = True):
        super(BriscolaRandomPlayer, self).__init__(PseudoRandomPlayer(), played=played)


class BriscolaEpsGreedyPlayer(BriscolaCustomEnemyPlayer):

    def __init__(self, eps: float = 0.2, played: bool = True):
        super(BriscolaEpsGreedyPlayer, self).__init__(EpsGreedyPlayer(eps, 1), played=played)
