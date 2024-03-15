from briscola_rl.players.base_player import BasePlayer


class HumanPlayer(BasePlayer):

    def choose_card(self) -> int:
        return -1
