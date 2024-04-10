import os

import numpy as np
import seaborn as sns
from gymnasium import spaces
from matplotlib import pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from game import BriscolaEpsGreedyPlayer, BriscolaRulesPlayer
from players.model_player import ModelPlayer
from state import PublicState


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

def read_model_stats():
    print(pd.read_csv('../data/epsgreedy_50000.csv')[['name', 'win_rate']])


def q_values(model, state: PublicState, env):
    obs = spaces.flatten(env.observation_space_nested, state.as_dict(played=False))
    t, _ = model.policy.obs_to_tensor(obs)
    return model.q_net(t).detach().numpy().squeeze()


def print_actions_for_state(hand, table, briscola):
    env = BriscolaRulesPlayer(played=False)
    model = DQN.load('../checkpoints/graceful-darkness', env)
    state = PublicState(0, 0, hand, 2, 34, table, [], [], 1, briscola, 1)
    plt.title("Briscola: " + briscola.__repr__() + "\nOpponent played: " + table[0].__repr__())
    sns.barplot(dict(
        x=[c.__repr__() for c in hand],
        y=q_values(model, state, env)
    ), x='x', y='y')
    plt.xlabel("Hand")
    plt.ylabel("Q value")
    plt.grid(True)
    plt.show()


def best_bar_plot():
    df = pd.concat([
        pd.read_csv('../data/random_best.csv'),
        pd.read_csv('../data/greedy_best.csv'),
        pd.read_csv('../data/rules_best.csv'),
    ])
    df.rename(columns={
        'strategy': 'Agent strategy',
        'wins_len_std': 'W.Ep. length stddev',
        'losses_len_std': 'L.Ep. length stddev',
        'rew_std': 'Ep. reward stddev',
    }, inplace=True)
    df['model'] = df['model'].str.replace('best_model', 'DQN selfplay')
    plt.figure(figsize=(14, 14))
    plt.axvline(x=0.5, linestyle='--', color='black')
    plt.title('Best models - Average win rate', weight='bold', fontsize=26)
    plt.xlabel("Win rate")
    ax = sns.barplot(df,
                x='wins_rate',
                y='model',
                hue='Agent strategy',
                order=df.groupby(['model'])['wins_rate'].mean().sort_values(ascending=False).index
                )
    sns.move_legend(ax, "lower right", bbox_to_anchor=(1, 1))
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:], l[:], fontsize='x-large', title='Agent strategy', title_fontsize='x-large')
    plt.ylabel('')
    plt.xlim([0, 1])
    plt.show()

def rules_results_stacked_bar_plot():
    df = pd.read_csv('../data/rules_best.csv')

    (df[['model', 'wins_rate', 'draws_rate', 'losses_rate']]
     .sort_values(by='wins_rate', ascending=True)
     .set_index('model')
     .plot(kind='barh', stacked=True, color=['green', 'blue', 'red']))
    plt.title('Stacked results')
    plt.figure(figsize=(12, 8))
    plt.axvline(x=0.5, linestyle='--', color='black')
    plt.show()

def lengths_rewards_correlations():
    df = pd.concat([
        pd.read_csv('../data/random_best.csv'),
        pd.read_csv('../data/greedy_best.csv'),
        pd.read_csv('../data/rules_best.csv'),
    ])
    df.rename(columns={
        'strategy': 'Agent strategy',
        'wins_len_std': 'W.Ep. length stddev',
        'losses_len_std': 'L.Ep. length stddev',
        'rew_std': 'Ep. reward stddev',
    }, inplace=True)
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(df,
                    x='wins_len_mean',
                    y='wins_rate',
                    hue='Agent strategy',
                    size='W.Ep. length stddev')
    plt.title('Win lengths & ratio - top is better', weight='bold', fontsize=18)
    plt.xlabel('Won episode length')
    plt.ylabel('Win rate')
    plt.xlim([14.5, 20])
    plt.ylim([0, 1])
    #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:], l[:], fontsize='small')
    plt.show()

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(df,
                    x='losses_len_mean',
                    y='losses_rate',
                    hue='Agent strategy',
                    size='L.Ep. length stddev')
    plt.title('Losses lengths & ratio - bottom is better', fontsize=18, weight='bold')
    plt.xlabel('Lost episode length')
    plt.ylabel('Loss rate')
    plt.xlim([14.5, 20])
    plt.ylim([0, 1])
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:], l[:], fontsize='small')
    plt.show()

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(df,
                    x='rew_mean',
                    y='wins_rate',
                    hue='Agent strategy',
                    size='Ep. reward stddev')
    plt.title('Mean reward & win ratio - top right is better', weight='bold', fontsize=18)
    plt.xlabel('Average reward per episode')
    plt.ylabel('Win rate')
    plt.xlim([-35, 39])
    plt.ylim([0, 1])
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:], l[:], fontsize='small')
    plt.show()

    #sns.lmplot(df,
    #                x='rew_mean',
    #                y='wins_rate',
    #                hue='strategy')
    #plt.title('Average reward against win rate - to the top right is better')
    #plt.show()

def played_vs_not_played_plot():
    df = pd.concat([
        pd.read_csv('../data/random_best.csv'),
        pd.read_csv('../data/greedy_best.csv'),
        pd.read_csv('../data/rules_best.csv'),
    ])
    df.rename(columns={
        'strategy': 'Agent strategy',
        'wins_len_std': 'W.Ep. length stddev',
        'losses_len_std': 'L.Ep. length stddev',
        'rew_std': 'Ep. reward stddev',
        'played': 'Played'
    }, inplace=True)
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(df,
                         x='wins_len_mean',
                         y='wins_rate',
                         hue='Played',
                         size='W.Ep. length stddev')
    plt.title('Win lengths & ratio - top is better', weight='bold', fontsize=18)
    plt.xlabel('Won episode length')
    plt.ylabel('Win rate')
    plt.xlim([14.5, 20])
    plt.ylim([0, 1])
    #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:], l[:], fontsize='small')
    plt.show()

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(df,
                         x='losses_len_mean',
                         y='losses_rate',
                         hue='Played',
                         size='L.Ep. length stddev')
    plt.title('Losses lengths & ratio - bottom is better', fontsize=18, weight='bold')
    plt.xlabel('Lost episode length')
    plt.ylabel('Loss rate')
    plt.xlim([14.5, 20])
    plt.ylim([0, 1])
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:], l[:], fontsize='small')
    plt.show()

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(df,
                         x='rew_mean',
                         y='wins_rate',
                         hue='Played',
                         size='Ep. reward stddev')
    plt.title('Mean reward & win ratio - top right is better', weight='bold', fontsize=18)
    plt.xlabel('Average reward per episode')
    plt.ylabel('Win rate')
    plt.xlim([-35, 39])
    plt.ylim([0, 1])
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:], l[:], fontsize='small')
    plt.show()



if __name__ == '__main__':
    sns.set_theme()
    # heatmap_vs_best_models(n_eval_episodes=100)
    # read_model_stats()
    # print_actions_for_state([Card(1, 1), Card(1, 2), Card(1, 3)], [Card(6, 2)], Card(2, 1))
    #best_bar_plot()
    #rules_results_stacked_bar_plot()
    #lengths_rewards_correlations()
    played_vs_not_played_plot()