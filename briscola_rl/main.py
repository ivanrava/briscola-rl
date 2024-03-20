import logging

import tqdm
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.env_checker import check_env

from briscola_rl.game import BriscolaRandomPlayer, BriscolaEpsGreedyPlayer


def test_match(number_of_rounds: int = 10_000):
    logging.basicConfig(filename='test_match.log', level=logging.INFO)
    logger = logging.getLogger('test_match')
    total_reward = 0
    wins = 0
    draws = 0
    losses = 0
    for _ in tqdm.tqdm(range(number_of_rounds)):
        reward = test_round(logger)
        total_reward += reward

        if reward > 0:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1
    return (
        total_reward / number_of_rounds,
        wins / number_of_rounds,
        draws / number_of_rounds,
        losses / number_of_rounds
    )


def test_round(logger: logging.Logger) -> float:
    logger.info("Starting new round")
    obs, _ = env.reset()
    total_reward = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, finisheds, _, info = env.step(action)
        total_reward += rewards
        if finisheds:
            logger.info("Finished round")
            return total_reward


if __name__ == '__main__':
    env = BriscolaRandomPlayer()
    check_env(env, warn=True)

    model = PPO('MlpPolicy', env, verbose=True)
    model.learn(total_timesteps=100_000, progress_bar=True)

    print(test_match())
