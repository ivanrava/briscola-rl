from stable_baselines3 import PPO, DQN, A2C

import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from briscola_rl.evaluate import test_match
from briscola_rl.game import BriscolaRandomPlayer, BriscolaEpsGreedyPlayer

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 2_500_000,
    "opponent": "EpsGreedyPlayer",
    "eps": 0.03,
    "exploration_fraction": 0.8,
    "learning_rate": 0.01,
    "played": False
}
run = wandb.init(
    project="briscola-rl",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,
    save_code=False,
)

def make_env():
    if config["opponent"] == "RandomPlayer":
        env = BriscolaRandomPlayer(played=config['played'])
    else:
        env = BriscolaEpsGreedyPlayer(eps=config['eps'], played=config['played'])
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env

if __name__ == '__main__':
    env = make_env()
    model = DQN(config['policy_type'], env, verbose=True, tensorboard_log=f"runs/{run.id}", exploration_fraction=config['exploration_fraction'], learning_rate=config['learning_rate'])
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f'models/{run.id}',
            model_save_freq=1000,
            verbose=2
        )
    )
    run.finish()

    # print(test_match(env, model))
