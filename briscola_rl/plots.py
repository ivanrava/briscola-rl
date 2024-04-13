import hashlib
import os

import numpy as np
import seaborn as sns
from gymnasium import spaces
from matplotlib import pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from cards import Card
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
    obs = spaces.flatten(env.observation_space_nested, state.as_dict(played=env.played))
    t, _ = model.policy.obs_to_tensor(obs)
    return model.q_net(t).detach().numpy().squeeze()


def print_actions_for_state_subplots(hand, table, briscola, models):
    df = pd.read_csv('../data/rules_best.csv')

    f, axes = plt.subplots(4, 2, figsize=(9,9), sharex=True, sharey=True)

    plt.suptitle("Briscola: " + briscola.__repr__() + "\nOpponent played: " + table[0].__repr__())
    for i,model_name in enumerate(models):
        model = DQN.load(f'../checkpoints/{model_name}')
        in_features = model.policy.q_net.q_net[0].in_features
        played = in_features == 2999
        env = BriscolaRulesPlayer(played=played)
        model.env = env
        state = PublicState(0, 0, hand, 2, 34, table, [], [], 1, briscola, 1)

        axes.flat[i].set_title(model_name)
        colors = ['y'] * 3
        q_vals = q_values(model, state, env)
        colors[np.argmax(q_vals)] = 'r'
        sns.barplot(dict(
            x=[c.__repr__() for c in hand],
            y=q_vals,
        ), x='x', y='y', ax=axes.flat[i], palette=colors)
        axes.flat[i].set_xlabel('')
        axes.flat[i].set_ylabel('')
        plt.grid(True)

    plt.show()


def print_actions_for_state(hand, table, briscola, models):
    results = []
    x_hand = [c.__repr__() for c in hand]

    for model_name in models:
        env = BriscolaRulesPlayer(played=False)
        model = DQN.load(f'../checkpoints/{model_name}', env)
        state = PublicState(0, 0, hand, 2, 34, table, [], [], 1, briscola, 1)

        q_vals = q_values(model, state, env)

        for i in range(3):
            results.append({
                'model': model_name,
                'card': x_hand[i],
                'q_value': q_vals[i]
            })

    df = pd.DataFrame(results)
    #sns.barplot(df, x='model', y='q_value', hue='card')
    sns.heatmap(df.pivot(index="model", columns="card", values="q_value"), fmt=".2f", annot=True, cmap='mako')
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

def reinforce_plot():
    import dill

    pickles_folder = '../pickles'
    lst = []
    for filename in os.listdir(pickles_folder):
        filepath = os.path.join(pickles_folder, filename)
        dat = dill.load(open(filepath, 'rb'))
        if dat['num_iterations'] > 10000:
            continue

        for index, it in enumerate(range(0, dat['num_iterations'], dat['eval_interval'])):
            hasher = hashlib.sha1(filename.encode())
            lst.append({
                'run': hasher.hexdigest()[:4],
                'iteration': it,
                'sparse_reward': dat['sparse_reward'],
                'penalty': dat['penalty'],
                'dense_reward': dat['dense_reward'],
                'learning_rate': dat['learning_rate'],
                'fc_layer_params': dat['fc_layer_params'],
                'replay_buffer_capacity': dat['replay_buffer_capacity'],
                'return': dat['returns'][index]
            })
        #sns.lineplot(dat, y=dat['returns'], x=np.linspace(0, dat['num_iterations'], len(dat['returns'])), color=['b', 'g'][dat['sparse_reward']])
        # sns.regplot(dat, y=dat['returns'], x=np.linspace(0, dat['num_iterations'], len(dat['returns'])), scatter_kws={'s':4})

    df = pd.DataFrame(lst)

    plt.figure(figsize=(12,6))
    ax = sns.lineplot(df, y='return', x='iteration', hue='run')
    sns.move_legend(
        ax, loc="upper left", ncol=1, frameon=True, columnspacing=1, handletextpad=0.3, bbox_to_anchor=(1,1)
    )
    plt.title('REINFORCE runs - under 10000 iterations', weight='bold', fontsize=18)
    plt.xlabel('Iterations')
    plt.ylabel('Returns')
    plt.show()

    #plt.figure(figsize=(12,6))
    #ax = sns.lmplot(df, y='return', x='iteration', hue='sparse_reward', height=10, aspect=1.2)
    #plt.suptitle('REINFORCE runs - sparse reward', weight='bold', fontsize=18)
    #plt.xlabel('Iterations')
    #plt.ylabel('Returns')
    #plt.show()

    #plt.figure(figsize=(12,6))
    #ax = sns.lmplot(df, y='return', x='iteration', hue='dense_reward', height=10, aspect=1.2)
    #plt.suptitle('REINFORCE runs - dense reward', weight='bold', fontsize=18)
    #plt.xlabel('Iterations')
    #plt.ylabel('Returns')
    #plt.show()

    #plt.figure(figsize=(12,6))
    #ax = sns.lmplot(df, y='return', x='iteration', hue='penalty', height=10, aspect=1.2)
    #plt.suptitle('REINFORCE runs - penalty reward', weight='bold', fontsize=18)
    #plt.xlabel('Iterations')
    #plt.ylabel('Returns')
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

    #hand = [Card(4, 1), Card(1, 3), Card(1, 1)]
    #briscola = Card(2, 1)
    #table = [Card(3, 3)]
    #print_actions_for_state_subplots(
    #    hand,
    #    table,
    #    briscola,
    #    ['blooming-bird', 'true-star', 'graceful-darkness', 'autumn-night', 'laced-pond', 'snowy-shape', 'mild-aaldvark', 'lively-cosmos']
    #)
    #print_actions_for_state_subplots(
    #    hand,
    #    table,
    #    briscola,
    #    ['chocolate-cherry', 'upbeat-darkness', 'helpful-sound', 'cosmic-firebrand', 'atomic-dragon', 'hardy-galaxy', 'skilled-serenity', 'kind-surf']
    #)

    #best_bar_plot()
    #rules_results_stacked_bar_plot()
    #lengths_rewards_correlations()
    played_vs_not_played_plot()