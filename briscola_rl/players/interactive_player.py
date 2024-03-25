from briscola_rl.players.base_player import BasePlayer


class InteractivePlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        self.name = 'InteractivePlayer'

    def choose_card(self, _state) -> int:
        #print(f'[{_state.turn}] - Points: [{_state.my_points}, {_state.other_points}]')
        #if _state.order == 0:
        #    print('You are first.')
        #else:
        #    print(f'Opponent played "{_state.table}"')
        #print(f'Your hand: {self.hand}')
        return int(input(f'Enter your card index (briscola is {_state.briscola}): '))
