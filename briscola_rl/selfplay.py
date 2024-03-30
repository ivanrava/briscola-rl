import os
from random import randint

import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback

from shutil import copyfile # keep track of generations

from wandb.integration.sb3 import WandbCallback

from briscola_rl.game import BriscolaEpsGreedyPlayer, BriscolaCustomEnemyPlayer
from briscola_rl.players.base_player import BasePlayer
from state import PublicState


LOGDIR = "ppo_selfplay"

config = {
  "seed": 17,
  "policy_type": "MlpPolicy",
  "total_timesteps": int(1e9),
  "opponent": "SelfPlay",
  "eps": 0.03,
  "exploration_fraction": 0.8,
  "learning_rate": 1e-3,
  "played": False,
  "eval_freq": int(1e5),
  "eval_episodes": int(1e2),
  "best_threshold": 0.5
}
run = wandb.init(
  project="briscola-rl",
  config=config,
  sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
  monitor_gym=False,
  save_code=False,
)

class SelfPlayAgentOpponent(BasePlayer):
  def __init__(self, env_with_models):
    super().__init__()
    self.env = env_with_models

  def choose_card(self, state: PublicState) -> int:
    if self.env.best_model is None:
      return randint(0, len(self.hand) - 1) if len(self.hand) > 1 else 0
    else:
      action, _ = self.env.best_model.predict(self.env._get_obs(), deterministic=True)
      while action >= len(self.hand):
        action -= 1
      return action


class BriscolaSelfPlayEnv(BriscolaCustomEnemyPlayer):
  # wrapper over the normal single player env, but loads the best self play model
  def __init__(self):
    super(BriscolaSelfPlayEnv, self).__init__(other_player=SelfPlayAgentOpponent(env_with_models=self))
    self.best_model = None
    self.best_model_filename = None

  def reset(self, seed=None, **kwargs):
    # load model if it's there
    modellist = [f for f in os.listdir(LOGDIR) if f.startswith("history")]
    modellist.sort()
    if len(modellist) > 0:
      filename = os.path.join(LOGDIR, modellist[-1]) # the latest best model
      if filename != self.best_model_filename:
        print("loading model: ", filename)
        self.best_model_filename = filename
        if self.best_model is not None:
          del self.best_model
        self.best_model = PPO.load(filename, env=self)
    return super(BriscolaSelfPlayEnv, self).reset(seed=seed)

class SelfPlayCallback(EvalCallback):
  # hacked it to only save new version of best model if beats prev self by BEST_THRESHOLD score
  # after saving model, resets the best score to be BEST_THRESHOLD
  def __init__(self, *args, **kwargs):
    super(SelfPlayCallback, self).__init__(*args, **kwargs)
    self.best_mean_reward = config['best_threshold']
    self.generation = 0
  def _on_step(self) -> bool:
    result = super(SelfPlayCallback, self)._on_step()
    if result and self.best_mean_reward > config['best_threshold']:
      self.generation += 1
      print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
      print("SELFPLAY: new best model, bumping up generation to", self.generation)
      source_file = os.path.join(LOGDIR, "best_model.zip")
      backup_file = os.path.join(LOGDIR, "history_"+str(self.generation).zfill(8)+".zip")
      copyfile(source_file, backup_file)
      self.best_mean_reward = config['best_threshold']
    return result

class GreedyEvalCallback(EvalCallback):
  def __init__(self, *args, **kwargs):
    super(EvalCallback, self).__init__(*args, **kwargs)

  def _on_rollout_end(self) -> None:
    mean_reward = evaluate_policy(self.model, BriscolaEpsGreedyPlayer(), n_eval_episodes=config['eval_episodes'])
    self.logger.record('eval_greedy_agent_mean_reward', mean_reward)


#def rollout(env, policy):
#  """ play one agent vs the other in modified gym-style loop. """
#  obs = env.reset()
#
#  done = False
#  total_reward = 0
#
#  while not done:
#    action, _states = policy.predict(obs)
#    obs, reward, done, _, _ = env.step(action)
#    total_reward += reward
#
#  return total_reward

def train():
  # train selfplay agent
  logger.configure(folder=LOGDIR)

  env = BriscolaSelfPlayEnv()
  env.reset(seed=config['seed'])

  #model = PPO('MlpPolicy', env, ent_coef=0.0, n_epochs=10, batch_size=64, gamma=0.99, verbose=2)
  model = PPO('MlpPolicy', env, verbose=2)

  eval_callback = SelfPlayCallback(env,
    best_model_save_path=LOGDIR,
    log_path=LOGDIR,
    eval_freq=config['eval_freq'],
    n_eval_episodes=config['eval_episodes'],
    deterministic=False
  )

  model.learn(total_timesteps=config['total_timesteps'], callback=[
    eval_callback,
    EvalCallback(
      eval_env=BriscolaEpsGreedyPlayer(eps=config['eps']),
      eval_freq=config['eval_freq'],
      n_eval_episodes=config['eval_episodes'],
      deterministic=True
    ),
    WandbCallback(
            gradient_save_freq=100,
            model_save_path=f'models/{run.id}',
            model_save_freq=1000,
            verbose=2
        )
  ])

  model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

  env.close()

if __name__ == "__main__":
  train()