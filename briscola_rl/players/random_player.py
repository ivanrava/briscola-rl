from briscola_rl.players.base_player import BasePlayer
from random import randint


class PseudoRandomPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        self.name = 'RandomPlayer'

    def choose_card(self) -> int:
        return randint(0, len(self.hand) - 1) if len(self.hand) > 1 else 0
