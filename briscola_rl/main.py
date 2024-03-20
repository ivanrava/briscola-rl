from stable_baselines3 import PPO, DQN, A2C

import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from briscola_rl.evaluate import test_match
from briscola_rl.game import BriscolaRandomPlayer, BriscolaEpsGreedyPlayer

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 50_000,
    "opponent": "EpsGreedyPlayer"
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
        env = BriscolaRandomPlayer()
    else:
        env = BriscolaEpsGreedyPlayer()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env

if __name__ == '__main__':
    env = make_env()
    model = PPO(config['policy_type'], env, verbose=True, tensorboard_log=f"runs/{run.id}")
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
