import logging
import os

import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from tqdm import tqdm

from game import BriscolaEpsGreedyPlayer


def rizzo_match(env, model, number_of_rounds: int = 10_000):
    logging.basicConfig(filename='test_match.log', level=logging.INFO)
    logger = logging.getLogger('test_match')
    total_reward = 0
    wins = 0
    draws = 0
    losses = 0
    for _ in tqdm(range(number_of_rounds)):
        reward = rizzo_round(env, model, logger)
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


def rizzo_round(env, model, logger: logging.Logger) -> float:
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


def evaluate_checkpoints(n_eval_episodes=10_000, seed=1337):
    checkpoints_folder = '../checkpoints'
    print(f'Results averaged over {n_eval_episodes} episodes:')

    def episode_length(predicate):
        good_lengths = [l for r, l in zip(rewards, lengths) if predicate(r)]
        return f'Length: {round(np.mean(good_lengths), 3):>6} +/- {round(np.std(good_lengths), 3)}'

    for filename in os.listdir(checkpoints_folder):
        filename = filename.rstrip('.zip')
        filepath = os.path.join(checkpoints_folder, filename)
        model = DQN.load(filepath)
        env = BriscolaEpsGreedyPlayer(played=False, eps=0.03)
        env.reset(seed=seed)
        env = Monitor(env)
        rewards, lengths = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True)
        print(f'{filename:>20}: {np.mean(rewards):<6} +/- {np.std(rewards)}')
        print(f'{"Wins":>20}: {len([r for r in rewards if r > 0])/len(rewards):<6} \t {episode_length(lambda r: r > 0)}')
        print(f'{"Draws":>20}: {len([r for r in rewards if r == 0])/len(rewards):<6} \t {episode_length(lambda r: r == 0)}')
        print(f'{"Losses":>20}: {len([r for r in rewards if r < 0])/len(rewards):<6} \t {episode_length(lambda r: r < 0)}')
        print()

if __name__ == '__main__':
    evaluate_checkpoints()