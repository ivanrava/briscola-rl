import os
from typing import Tuple, Optional, Union, Dict

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import PolicyPredictor
import pandas as pd

from game import BriscolaRulesPlayer, BriscolaRandomPlayer, BriscolaEpsGreedyPlayer


def q_values(m, obs):
    t, _ = m.policy.obs_to_tensor(obs)
    return m.q_net(t).detach().cpu().numpy().squeeze()


class MetaModel(PolicyPredictor):
    CHECKPOINTS_FOLDER = '../checkpoints'

    def __init__(self, voting_strategy='voting', whitelist=('true-star', 'blooming-bird')):
        self.whitelist = whitelist

        self.played_names = []
        self.not_played_names = []
        self.models_played, self.models_not_played = self._hydrate_models()
        self.voting_strategy = voting_strategy
        df = pd.read_csv('../data/rules_best.csv')
        df_grouped = df.groupby(['model']).mean().sort_values('wins_rate')[['wins_rate']]
        self.weights = df_grouped / df_grouped.sum()

    def _weight(self, model_name: str):
        return float(self.weights.loc[model_name].iloc[0])


    def _hydrate_models(self):
        models_played = []
        models_not_played = []
        for filename in os.listdir(self.CHECKPOINTS_FOLDER):
            filename = filename.rstrip('.zip')
            if filename not in self.whitelist:
                continue
            filepath = os.path.join(self.CHECKPOINTS_FOLDER, filename)
            m = DQN.load(filepath)
            in_features = m.policy.q_net.q_net[0].in_features
            played = in_features == 2999
            models_played.append(m) if played else models_not_played.append(m)
            self.played_names.append(filename) if played else self.not_played_names.append(filename)
        return models_played, models_not_played

    def predict(self,
                observation: Union[np.ndarray, Dict[str, np.ndarray]],
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = True,
                ):
        if self.voting_strategy == 'voting':
            return self._predict_voting(observation, state, episode_start, deterministic, weighted=False)
        elif self.voting_strategy == 'voting_weighted':
            return self._predict_voting(observation, state, episode_start, deterministic, weighted=True)
        elif self.voting_strategy == 'q_raw':
            return self._predict_q_voting(observation)
        elif self.voting_strategy == 'q_normal':
            return self._predict_q_voting(observation, normalize=True)
        else:
            raise NotImplementedError(f"Voting strategy {self.voting_strategy} unknown")

    def _predict_voting(self,
                        observation: Union[np.ndarray, Dict[str, np.ndarray]],
                        state: Optional[Tuple[np.ndarray, ...]] = None,
                        episode_start: Optional[np.ndarray] = None,
                        deterministic: bool = True,
                        weighted: bool = False,
                        ):
        obs_played = observation

        size_points_oh = 31*40
        start_of_my_played = 31 + 31*3 + 4
        end_of_my_played = start_of_my_played + size_points_oh
        start_of_other_played = start_of_my_played + size_points_oh + 1 + 4
        end_of_other_played = start_of_other_played + size_points_oh
        obs_not_played = np.concatenate([
            observation.squeeze()[:start_of_my_played],
            observation.squeeze()[end_of_my_played:start_of_other_played],
            observation.squeeze()[end_of_other_played:]
        ])
        obs_not_played = np.array(obs_not_played)

        actions = [0,0,0]
        for n, m in zip(self.not_played_names, self.models_not_played):
            action, _ = m.predict(observation=obs_not_played, state=state, episode_start=episode_start, deterministic=True)
            actions[action] += self._weight(n) if weighted else 1
        for n, m in zip(self.played_names, self.models_played):
            action, _ = m.predict(observation=obs_played, state=state, episode_start=episode_start, deterministic=True)
            actions[action[0]] += self._weight(n) if weighted else 1
        return [np.argmax(actions)], None

    def _predict_q_voting(self,
                        observation: Union[np.ndarray, Dict[str, np.ndarray]],
                        state: Optional[Tuple[np.ndarray, ...]] = None,
                        episode_start: Optional[np.ndarray] = None,
                        deterministic: bool = True,
                        normalize: bool = False
                        ):
        obs_played = observation
        obs_not_played = np.array([observation.squeeze()[:-(31*80)]])
        actions = [0,0,0]
        for m in self.models_not_played:
            q_vals = q_values(m, obs_not_played)
            actions += (q_vals-np.min(q_vals))/(np.max(q_vals)-np.min(q_vals)) if normalize else q_vals
        for m in self.models_played:
            q_vals = q_values(m, obs_played)
            actions += (q_vals-np.min(q_vals))/(np.max(q_vals)-np.min(q_vals)) if normalize else q_vals
        return [np.argmax(actions)], None

if __name__ == '__main__':
    def episode_length(predicate):
        good_lengths = [l for r, l in zip(rewards, lengths) if predicate(r)]
        return f'Length: {round(np.mean(good_lengths) if len(good_lengths) > 0 else 0, 3):>6} +/- {round(np.std(good_lengths) if len(good_lengths) > 0 else 0, 3)}'

    model = MetaModel(voting_strategy='voting', whitelist=(
        'true-star', 'blooming-bird', 'graceful-darkness', 'autumn-night', 'laced-pond', 'snowy-shape', 'mild-aaldvark', 'lively-cosmos',
        'bumbling-leaf', 'easy-shape', 'earnest-night', 'rich-mountain', 'amber-lake'
    ))
    env = BriscolaRulesPlayer(played=True)
    env.reset(seed=1337)
    env = Monitor(env)
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=10_000, return_episode_rewards=True)
    print(f': {np.mean(rewards):<6} +/- {np.std(rewards)}')
    print(f'{"Wins":>20}: {len([r for r in rewards if r > 0])/len(rewards):<6} \t {episode_length(lambda r: r > 0)}')
    print(f'{"Draws":>20}: {len([r for r in rewards if r == 0])/len(rewards):<6} \t {episode_length(lambda r: r == 0)}')
    print(f'{"Losses":>20}: {len([r for r in rewards if r < 0])/len(rewards):<6} \t {episode_length(lambda r: r < 0)}')
    print()

    model = MetaModel(voting_strategy='voting_weighted', whitelist=(
        'true-star', 'blooming-bird', 'graceful-darkness', 'autumn-night', 'laced-pond', 'snowy-shape', 'mild-aaldvark', 'lively-cosmos',
        'bumbling-leaf', 'easy-shape', 'earnest-night', 'rich-mountain', 'amber-lake'
    ))
    env = BriscolaRulesPlayer(played=True)
    env.reset(seed=1337)
    env = Monitor(env)
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=10_000, return_episode_rewards=True)
    print(f': {np.mean(rewards):<6} +/- {np.std(rewards)}')
    print(f'{"Wins":>20}: {len([r for r in rewards if r > 0])/len(rewards):<6} \t {episode_length(lambda r: r > 0)}')
    print(f'{"Draws":>20}: {len([r for r in rewards if r == 0])/len(rewards):<6} \t {episode_length(lambda r: r == 0)}')
    print(f'{"Losses":>20}: {len([r for r in rewards if r < 0])/len(rewards):<6} \t {episode_length(lambda r: r < 0)}')
    print()

    model = MetaModel(voting_strategy='q_raw', whitelist=(
        'true-star', 'blooming-bird', 'graceful-darkness', 'autumn-night', 'laced-pond', 'snowy-shape', 'mild-aaldvark', 'lively-cosmos',
        'bumbling-leaf', 'easy-shape', 'earnest-night', 'rich-mountain', 'amber-lake'
    ))
    env = BriscolaRulesPlayer(played=True)
    env.reset(seed=1337)
    env = Monitor(env)
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=10_000, return_episode_rewards=True)
    print(f': {np.mean(rewards):<6} +/- {np.std(rewards)}')
    print(f'{"Wins":>20}: {len([r for r in rewards if r > 0])/len(rewards):<6} \t {episode_length(lambda r: r > 0)}')
    print(f'{"Draws":>20}: {len([r for r in rewards if r == 0])/len(rewards):<6} \t {episode_length(lambda r: r == 0)}')
    print(f'{"Losses":>20}: {len([r for r in rewards if r < 0])/len(rewards):<6} \t {episode_length(lambda r: r < 0)}')
    print()

    model = MetaModel(voting_strategy='q_normal', whitelist=(
        'true-star', 'blooming-bird', 'graceful-darkness', 'autumn-night', 'laced-pond', 'snowy-shape', 'mild-aaldvark',
        'lively-cosmos',
        'bumbling-leaf', 'easy-shape', 'earnest-night', 'rich-mountain', 'amber-lake'
    ))
    env = BriscolaRulesPlayer(played=True)
    env.reset(seed=1337)
    env = Monitor(env)
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=10_000, return_episode_rewards=True)
    print(f': {np.mean(rewards):<6} +/- {np.std(rewards)}')
    print(f'{"Wins":>20}: {len([r for r in rewards if r > 0]) / len(rewards):<6} \t {episode_length(lambda r: r > 0)}')
    print(f'{"Draws":>20}: {len([r for r in rewards if r == 0]) / len(rewards):<6} \t {episode_length(lambda r: r == 0)}')
    print(f'{"Losses":>20}: {len([r for r in rewards if r < 0]) / len(rewards):<6} \t {episode_length(lambda r: r < 0)}')
    print()
