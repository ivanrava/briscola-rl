from briscola_rl.players.base_player import BasePlayer


class HumanPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        self.name = 'HumanPlayer'

    def choose_card(self) -> int:
        return -1
