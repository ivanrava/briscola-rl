import logging
import os
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy

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
        reward = test_round(env, model, logger)
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

if __name__ == '__main__':
    cartella = '../checkpoints'

    # Iterazione sui file nella cartella
    for nome_file in os.listdir(cartella):
        nome_file = nome_file.rstrip('.zip')
        percorso_file = os.path.join(cartella, nome_file)
        model = DQN.load(percorso_file)
        print(nome_file)
        print(f'{evaluate_policy(model, BriscolaEpsGreedyPlayer(played=False, eps=0.03), n_eval_episodes=1000)}\n')