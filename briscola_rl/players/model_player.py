from gymnasium import spaces

from players.base_player import BasePlayer
from state import PublicState


class ModelPlayer(BasePlayer):
    def __init__(self, model, env):
        super().__init__()
        self.model = model
        self.env = env

    def choose_card(self, state: PublicState) -> int:
        state_copy = state

        state_copy.my_points, state_copy.other_points = state_copy.other_points, state_copy.my_points
        state_copy.other_hand_size = len(state_copy.hand)
        state_copy.hand = self.hand
        state_copy.order = 1 - state_copy.order
        state_copy.my_played, state_copy.other_played = state_copy.other_played, state_copy.my_played

        obs = spaces.flatten(self.env.observation_space_nested, state.as_dict(played=self.env.played))
        action, _ = self.model.predict(obs)
        while action >= len(self.hand):
            action = action - 1
        return action