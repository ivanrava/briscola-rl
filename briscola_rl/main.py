import numpy as np
from stable_baselines3 import PPO, DQN, A2C

import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from gymnasium import spaces

from briscola_rl.evaluate import test_match
from briscola_rl.game import BriscolaRandomPlayer, BriscolaEpsGreedyPlayer

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 2_500_000,
    "opponent": "EpsGreedyPlayer",
    "eps": 0.03,
    "exploration_fraction": 0.8,
    "learning_rate": 0.03,
    "played": False,
    "big_reward": False
}
run = wandb.init(
    project="briscola-rl",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,
    save_code=False,
)


class WinRateCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(WinRateCallback, self).__init__(verbose)
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.matches = 0

    def _on_step(self) -> bool:
        state = self.locals['infos'][-1].get('terminal_observation')
        if state is not None:
            self.matches += 1
            state_decoded = spaces.unflatten(self.training_env.get_attr('observation_space_nested')[-1], state)
            my_points = state_decoded['my_points']
            other_points = state_decoded['other_points']
            if my_points > other_points:
                self.wins += 1
            elif other_points > my_points:
                self.losses += 1
            elif my_points == other_points:
                self.draws += 1

        return True

    def _on_rollout_end(self) -> None:

        self.logger.record('win_rate', self.wins/self.matches)
        self.logger.record('draw_rate', self.draws/self.matches)
        self.logger.record('loss_rate', self.losses/self.matches)

        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.matches = 0


def make_env():
    if config["opponent"] == "RandomPlayer":
        env = BriscolaRandomPlayer(played=config['played'], big_reward=config['big_reward'])
    else:
        env = BriscolaEpsGreedyPlayer(eps=config['eps'], played=config['played'], big_reward=config['big_reward'])
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env


if __name__ == '__main__':
    env = make_env()
    model = DQN(config['policy_type'], env, verbose=True, tensorboard_log=f"runs/{run.id}",
                exploration_fraction=config['exploration_fraction'], learning_rate=config['learning_rate'], train_freq=(4, 'episode'))
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=[WandbCallback(
            gradient_save_freq=100,
            model_save_path=f'models/{run.id}',
            model_save_freq=1000,
            verbose=2
        ), WinRateCallback()]
    )
    run.finish()

    # print(test_match(env, model))
