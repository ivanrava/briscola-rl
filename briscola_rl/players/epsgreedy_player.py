from briscola_rl.players.base_player import BasePlayer
from random import randint, random
from briscola_rl.game_rules import select_winner
from state import PublicState


class EpsGreedyPlayer(BasePlayer):
    def __init__(self, epsilon, order_idx: int):
        """
        :param epsilon: controls exploration-exploitation tradeoff (high epsilon, more exploration).
        Should be low (e.g. 0.1).
        """
        super().__init__()
        self.epsilon = epsilon
        self.name = 'EpsGreedyPlayer'
        self.order_idx = order_idx

    def choose_card(self, state: PublicState) -> int:
        if self.epsilon > random():
            # Exploration
            return randint(0, len(self.hand) - 1) if len(self.hand) > 1 else 0
        # Exploitation
        return self.greedy_action(state)

    def greedy_action(self, state: PublicState):
        im_first = state.order == self.order_idx
        if im_first:
            return self.card_min_points(state)
        else:
            return self.card_max_gain(state)

    def card_max_gain(self, state: PublicState):
        i_max = -1
        max_gain = -100
        i_max_b = -1
        max_gain_b = -100
        table = state.table[:]
        table.append(None)
        for i, c in enumerate(self.hand):
            table[-1] = c
            winner = select_winner(table, state.briscola)
            coef_pts = 1 if winner else -1
            gain = coef_pts * sum(map(lambda card: card.points, table))
            if c.suit != state.briscola.suit:
                if gain > max_gain:
                    i_max = i
                    max_gain = gain
            else:
                if gain > max_gain_b:
                    i_max_b = i
                    max_gain_b = gain
        return i_max if max_gain >= 0 or max_gain_b < 0 else i_max_b

    def card_min_points(self, state):
        # TODO: write a better code, I'm triggered
        i_min = -1
        min_pts = 1000
        i_min_b = -1
        min_pts_b = 1000
        for i, c in enumerate(self.hand):
            if c.suit != state.briscola.suit:
                if c.points < min_pts:
                    i_min = i
                    min_pts = c.points
                else:
                    i_min_b = i
                    min_pts_b = c.points

        if min_pts <= 4:
            return i_min
        elif min_pts_b < 4 < min_pts:
            return i_min_b
        else:
            return i_min
