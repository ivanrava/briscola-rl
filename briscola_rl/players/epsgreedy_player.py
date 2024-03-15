from briscola_rl.players.base_player import BasePlayer
from random import randint, random
from briscola_rl.game_rules import select_winner


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

    def choose_card(self) -> int:
        if self.epsilon > random():
            # Exploration
            return randint(0, len(self.hand) - 1) if len(self.hand) > 1 else 0
        # Exploitation
        return self.greedy_action()

    def greedy_action(self):
        im_first = self.get_public_state().order == self.order_idx
        if im_first:
            return self.card_min_points()
        else:
            return self.card_max_gain()

    def card_max_gain(self):
        i_max = -1
        max_gain = -100
        state = self.get_public_state()
        table = state.table[:]
        table.append(None)
        for i, c in enumerate(self.hand):
            table[-1] = c
            winner = select_winner(table, state.briscola)
            coef_pts = 1 if winner else -1
            gain = coef_pts * sum(map(lambda card: card.points, table))
            if gain > max_gain:
                i_max = i
                max_gain = gain
        return i_max

    def card_min_points(self):
        # TODO: write a better code, I'm triggered
        i_min = -1
        min_pts = 1000
        for i, c in enumerate(self.hand):
            if c.points < min_pts:
                i_min = i
                min_pts = c.points
        return i_min
