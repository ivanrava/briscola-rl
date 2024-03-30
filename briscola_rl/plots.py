import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from game import BriscolaEpsGreedyPlayer
from players.model_player import ModelPlayer


def heatmap_vs_best_models(n_eval_episodes=5, seed=1337):
    results = []

    def episode_lengths(predicate, prefix):
        good_lengths = [l for r, l in zip(rewards, lengths) if predicate(r)]
        return {
            f'{prefix}_len_mean': round(np.mean(good_lengths), 3),
            f'{prefix}_len_std': round(np.std(good_lengths), 3)
        }

    checkpoints_folder = '../checkpoints'
    for filename_champion in os.listdir(checkpoints_folder):
        for filename_challenger in os.listdir(checkpoints_folder):
            filename_champion = filename_champion.rstrip('.zip')
            filepath = os.path.join(checkpoints_folder, filename_champion)
            champion_model = DQN.load(filepath)
            filename_challenger = filename_challenger.rstrip('.zip')
            filepath = os.path.join(checkpoints_folder, filename_challenger)
            challenger_model = DQN.load(filepath)
            env = BriscolaEpsGreedyPlayer(played=False, eps=0.03)
            env.other_player = ModelPlayer(challenger_model, env)
            env.reset(seed=seed)
            env = Monitor(env)
            rewards, lengths = evaluate_policy(champion_model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True)

            results.append({
                'champion': filename_champion,
                'challenger': filename_challenger,
                'rew_mean': np.mean(rewards),
                'rew_std': np.std(rewards),
                'wins_rate': len([r for r in rewards if r > 0])/len(rewards),
                'draws_rate': len([r for r in rewards if r == 0])/len(rewards),
                'losses_rate': len([r for r in rewards if r < 0])/len(rewards),
            } | episode_lengths(lambda r: r > 0, 'wins')
              | episode_lengths(lambda r: r == 0, 'draws')
              | episode_lengths(lambda r: r < 0, 'losses')
            )

    df = pd.DataFrame(results)
    print(df)
    plt.figure(figsize=(10, 8))
    plt.title("Win rate")
    plt.xlabel("Challenger")
    plt.ylabel("Champion")
    sns.heatmap(df.pivot(index="champion", columns="challenger", values="wins_rate"), fmt=".2f", annot=True, cmap='mako')
    plt.xticks(rotation=45)
    plt.show()


if __name__ == '__main__':
    heatmap_vs_best_models(n_eval_episodes=100)