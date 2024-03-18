from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.env_checker import check_env

from briscola_rl.game import BriscolaRandomPlayer

if __name__ == '__main__':
    env = BriscolaRandomPlayer()
    check_env(env, warn=True)

    model = PPO('MlpPolicy', env, verbose=True)
    model.learn(total_timesteps=10000)

    obs, _ = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, finisheds, _, info = env.step(action)
