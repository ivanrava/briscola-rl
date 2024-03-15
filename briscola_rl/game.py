import logging
from random import randint

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
    def __init__(self, other_player: BasePlayer):
        self.action_space = spaces.Discrete(3)  # drop i-th card
        self.my_player: BasePlayer = HumanPlayer()
        self.other_player = other_player
        self.players = [self.my_player, self.other_player]
        self.reward_range = (-22, 22)
        card_space = spaces.Tuple((spaces.Discrete(11), spaces.Discrete(5), spaces.Discrete(12)))  # (value, seed, points)
        self.observation_space = spaces.Dict({
            'my_points': spaces.Discrete(121),
            'other_points': spaces.Discrete(121),
            'hand_size': spaces.Discrete(4),
            'other_hand_size': spaces.Discrete(4),
            'remaining_deck_cards': spaces.Discrete(41),
            'hand': spaces.Tuple([card_space, card_space, card_space]),
            'table': spaces.Tuple([card_space, card_space]),
            'my_taken': spaces.Tuple([card_space] * 40),
            'other_taken': spaces.Tuple([card_space] * 40),
            'turn': spaces.Discrete(40),
            'briscola': card_space,
            'order': spaces.Discrete(2)
        })
        self.deck = None
        self.briscola: Card = None
        self.__logger = logging.getLogger('Briscola')
        self.turn_my_player = 0 # I start

    def step(self, action):
        assert action in self.action_space
        self.turn += 1
        my_card = self.my_player.hand.pop(action)
        self.table.append(my_card)
        if self.turn_my_player == 0:
            other_card = self.other_player.play_card()
            self.table.append(other_card)
        self.__logger.info(f'Table: {self.table}')
        # i_winner: 0 -> I win, 1 -> Enemy wins :(
        i_winner = select_winner(self.table, self.briscola)
        reward = self._state_update_after_winner(i_winner)
        self._draw_phase()
        if self.turn_my_player == 1:
            other_card = self.other_player.play_card()
            self.table.append(other_card)
        return self.public_state().as_dict(), reward, self.is_terminated(), False, dict()

    def _state_update_after_winner(self, i_winner: int):
        """
        :param i_winner: 0 -> I win, 1 -> I lose
        :return: The reward
        """
        self.__logger.info(f'Turn Winner is {self.players[i_winner].name}')
        reward = gained_points = sum(values_points[c.value] for c in self.table)
        self.points[0] += gained_points
        self.my_taken.append(self.table[self.turn_my_player])
        self.other_taken.append(self.table[1 - self.turn_my_player])
        gained_points_my_player = gained_points_other_player = gained_points
        if i_winner == self.turn_my_player:
            self.my_points += gained_points
            self.turn_my_player = 0
            gained_points_other_player = gained_points_other_player * -1
        else:
            self.other_points += gained_points
            self.turn_my_player = 1
            gained_points_my_player = gained_points_my_player * -1
            reward = reward * -1
        self.my_player.notify_turn_winner(gained_points_my_player)
        self.other_player.notify_turn_winner(gained_points_other_player)
        self.table = []
        self.__logger.info(f'Winner gained {gained_points} points')
        self.__logger.info(f'Current table points: {self.points}')
        return reward

    def _draw_phase(self):
        if not self.deck.is_empty():
            c1 = self.deck.draw()
            c2 = self.deck.draw()
            if self.turn_my_player == 0:
                c_my_player = c1
                c_other_player = c2
            else:
                c_other_player = c1
                c_my_player = c2
            self.my_player.hand.append(c_my_player)
            self.other_player.hand.append(c_other_player)

    def public_state(self):
        return PublicState(self.my_points, self.other_points, self.my_player.hand,
                           len(self.other_player.hand), len(self.deck.cards),
                           self.table, self.my_taken, self.other_taken,
                           self.turn, self.briscola, self.turn_my_player)

    def is_terminated(self):
        return any(p > 60 for p in self.points) or \
               (self.deck.is_empty() and all(len(p.hand) == 0 for p in self.players))

    def reset(self, **kwargs):
        self.turn = 0
        self.my_player.reset_player()
        self.other_player.reset_player()
        self.deck = Deck()
        self.my_taken = []
        self.table = []
        self.other_taken = []
        self.my_points = 0
        self.other_points = 0
        self.points = [0, 0]
        self.turn_my_player = randint(0, 1)
        self.players = [self.my_player, self.other_player]
        self.briscola: Card = self.deck.draw()
        for _ in range(3):
            self.my_player.hand.append(self.deck.draw())
        for _ in range(3):
            self.other_player.hand.append(self.deck.draw())
        self.deck.cards.append(self.briscola)
        if self.turn_my_player == 1:
            other_card = self.other_player.play_card()
            self.table.append(other_card)

    def render(self, mode="human"):
        pass


class BriscolaRandomPlayer(BriscolaCustomEnemyPlayer):

    def __init__(self):
        super(BriscolaRandomPlayer, self).__init__(PseudoRandomPlayer())


class BriscolaEpsGreedyPlayer(BriscolaCustomEnemyPlayer):

    def __init__(self, eps: float = 0.2):
        super().__init__(EpsGreedyPlayer(eps, 1))