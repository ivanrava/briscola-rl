from briscola_rl.players.base_player import BasePlayer
from random import randint, random
from briscola_rl.game_rules import select_winner
from cards import Card
from state import PublicState


class RulesPlayer(BasePlayer):
    def __init__(self, order_idx: int):
        super().__init__()
        self.name = 'RulesPlayer'
        self.order_idx = order_idx

    def choose_card(self, state: PublicState) -> int:
        return self.smart_action(state)

    def smart_action(self, state: PublicState):
        im_first = state.order == self.order_idx
        if im_first:
            return self.first_move(state)
        else:
            return self.second_move(state)


    def win_move(self, state: PublicState):
        table = state.table[:]
        table.append(None)
        for i,c in enumerate(self.hand):
            table[-1] = c
            winner = select_winner(table, state.briscola)
            coef_pts = 1 if winner else -1
            gain = coef_pts * sum(map(lambda card: card.points, table))
            if state.my_points + gain > 60:
                return i
        return None

    def filter_losses_from_hand(self, state: PublicState):
        table = state.table[:]
        table.append(None)
        winnable_hand = []
        drawable_hand = []
        for i,c in enumerate(self.hand):
            table[-1] = c
            winner = select_winner(table, state.briscola)
            coef_pts = 1 if winner else -1
            gain = coef_pts * sum(map(lambda card: card.points, table))
            if state.other_points + -1 * gain < 60:
                winnable_hand.append(i)
            if state.other_points + -1 * gain <= 60:
                drawable_hand.append(i)

        assert len(drawable_hand) >= len(winnable_hand)
        return winnable_hand, drawable_hand

    def strozzo(self, state: PublicState):
        table = state.table[:]
        table.append(None)
        if table[0].suit == state.briscola.suit:
            return None

        max_gain = 0
        max_idx = None
        for i,c in enumerate(self.hand):
            table[-1] = c
            winner = select_winner(table, state.briscola)
            coef_pts = 1 if winner else -1
            gain = coef_pts * sum(map(lambda card: card.points, table))
            if gain > 0 and self.hand[i].suit == table[0].suit and gain > max_gain:
                max_gain = gain
                max_idx = i
        return max_idx

    def carico_response(self, state: PublicState):
        table = state.table[:]
        table.append(None)

        min_points_win_b = 100
        min_idx_win_b = None
        min_points_b = 100
        min_idx_b = None
        min_points = 100
        min_idx = None
        for i,c in enumerate(self.hand):
            table[-1] = c
            winner = select_winner(table, state.briscola)
            coef_pts = 1 if winner else -1
            gain = coef_pts * sum(map(lambda card: card.points, table))
            if c.suit == state.briscola.suit:
                if (min_idx_win_b is None or c.points < min_points_win_b) and gain > 0:
                    min_points_win_b = c.points
                    min_idx_win_b = i
                if min_idx_b is None or c.points < min_points_b:
                    min_points_b = c.points
                    min_idx_b = i
            else:
                if c.points < min_points or min_idx is None:
                    min_points = c.points
                    min_idx = i

        if min_idx_win_b is not None:
            return min_idx_win_b
        elif min_idx is not None:
            return min_idx
        else:
            return min_idx_b

    def nothing_response(self, state):
        # Table: small not briscola (not 3 or 1)
        min_points = 100
        min_idx = None
        min_loss = 100
        min_points_b = 100
        min_idx_b = None
        min_gain_b = 100

        table = state.table[:]
        table.append(None)
        for i,c in enumerate(self.hand):
            table[-1] = c
            winner = select_winner(table, state.briscola)
            coef_pts = 1 if winner else -1
            gain = coef_pts * sum(map(lambda card: card.points, table))
            if c.suit == state.briscola:
                if min_idx_b is None or c.points < min_points_b:
                    min_points_b = c.points
                    min_idx_b = i
                    min_gain_b = gain
            else:
                if min_idx is None or c.points < min_points:
                    min_points = c.points
                    min_idx = i
                    min_loss = gain

        if min_loss <= 3 or min_points_b >= 4:
            return min_idx
        else:
            play_briscola = random() > 0.8
            if play_briscola:
                return min_idx_b
            else:
                return min_idx


    def second_move(self, state: PublicState):
        win_idx = self.win_move(state)
        if win_idx is not None:
            return win_idx

        winnable_hand, drawable_hand = self.filter_losses_from_hand(state)
        if len(drawable_hand) == 0:
            # Lose :(
            return -1
        if len(winnable_hand) == 1:
            return winnable_hand[-1]
        if len(drawable_hand) == 1:
            return drawable_hand[-1]
        if len(winnable_hand) == 0:
            winnable_hand = drawable_hand

        strozzo_idx = self.strozzo(state)
        if strozzo_idx is not None:
            return strozzo_idx

        table = state.table[:]
        table.append(None)
        carico = table[0].points >= 10

        if carico:
            return self.carico_response(state)
        else:
            if table[0].suit == state.briscola.suit:
                return self.liscio(state)
            else:
                return self.nothing_response(state)

    def first_move(self, state):
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
                if c.points < min_pts_b:
                    i_min_b = i
                    min_pts_b = c.points

        if min_pts <= 4:
            return i_min
        elif min_pts_b < 4 < min_pts:
            return i_min_b
        else:
            return i_min

    def liscio(self, state):
        # Table: small briscola (not 3 or 1)
        table = state.table[:]
        table.append(None)

        min_points_b = 100
        min_idx_b = None
        min_points = 100
        min_idx = None
        min_gain_b = None
        min_loss = None
        for i,c in enumerate(self.hand):
            table[-1] = c
            winner = select_winner(table, state.briscola)
            coef_pts = 1 if winner else -1
            gain = coef_pts * sum(map(lambda card: card.points, table))
            if c.suit == state.briscola.suit:
                if min_idx_b is None or c.points < min_points_b:
                    min_points_b = c.points
                    min_idx_b = i
                    min_gain_b = gain
            else:
                if c.points < min_points or min_idx is None:
                    min_points = c.points
                    min_idx = i
                    min_loss = gain

        if min_idx is not None:
            return min_idx
        else:
            return min_idx_b


