from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.env_checker import check_env

from briscola_rl.game import BriscolaRandomPlayer, BriscolaEpsGreedyPlayer


def test_match(number_of_rounds: int = 1000) -> float:
    total_reward = 0
    for _ in range(number_of_rounds):
        reward = test_round()
        total_reward += reward
    return total_reward / number_of_rounds


def test_round() -> float:
    obs, _ = env.reset()
    total_reward = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, finisheds, _, info = env.step(action)
        total_reward += rewards
        if finisheds:
            return total_reward


if __name__ == '__main__':
    env = BriscolaRandomPlayer()
    check_env(env, warn=True)

    model = PPO('MlpPolicy', env, verbose=True)
    model.learn(total_timesteps=10000)

    print(test_match())
