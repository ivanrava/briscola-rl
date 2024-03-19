from briscola_rl.players.base_player import BasePlayer


class HumanPlayer(BasePlayer):
    name = 'HumanPlayer'

    def choose_card(self) -> int:
        return -1
