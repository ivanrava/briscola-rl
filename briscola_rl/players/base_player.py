from abc import ABC, abstractmethod
import logging
from briscola_rl.state import PublicState


class BasePlayer(ABC):
    def __init__(self, obs_public_state: PublicState = None):
        self.hand = []
        self.name = None
        self.__obs_public_state = obs_public_state
        self.__logger = logging.getLogger('Briscola')

    def reset_player(self):
        self.hand = []

    @abstractmethod
    def choose_card(self, state: PublicState) -> int:
        pass

    def play_card(self, state: PublicState):
        i = self.choose_card(state)
        try:
            c = self.hand.pop(i)
            self.__logger.info(f'{self.name} plays {c}')
            return c
        except IndexError as e:
            print('hand len: ', len(self.hand), 'i: ', i)
            raise e

    def on_enemy_play(self, card):
        pass

    def is_empty_hand(self):
        return len(self.hand) == 0

    def notify_turn_winner(self, points):
        pass

    def notify_game_winner(self, name: str):
        pass
