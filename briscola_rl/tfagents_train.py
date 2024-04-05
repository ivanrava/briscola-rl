import os
import tempfile

import numpy as np
import reverb
import tensorflow as tf
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.agents.sac import tanh_normal_projection_network, sac_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network, sequential
from tf_agents.policies import py_tf_eager_policy, random_tf_policy, random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.train import actor, triggers, learner
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from cards import Card, Deck
from game_rules import select_winner, values_points
from players.base_player import BasePlayer
from players.human_player import HumanPlayer
from players.rules_player import RulesPlayer
from state import PublicState


def flatten(data):
  return [y for x in data for y in (x if isinstance(x,list) or isinstance(x,tuple) else [x])]

class BriscolaTFEnv(py_environment.PyEnvironment):
  def __init__(self, sparse_reward: bool = False, dense_reward: bool = True, penalty: bool = False):
    super().__init__()
    self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
    self._observation_spec = array_spec.ArraySpec(shape=(25,), dtype=np.int64)

    self.sparse_reward = sparse_reward
    self.dense_reward = dense_reward
    self.penalty = penalty

    self.my_player: BasePlayer = HumanPlayer()
    self.other_player = RulesPlayer(1)
    self.players = [self.my_player, self.other_player]

    self._reset()
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def init_state(self):
    self.my_points = 0
    self.other_points = 0
    self.points = [self.my_points, self.other_points]
    self.turn = 0
    self.table = []
    self.deck = Deck(seed=np.random.random())
    self.my_played = []
    self.other_played = []
    self.my_player.hand = []
    for _ in range(3):
      self.my_player.hand.append(self.deck.draw())
    self.other_player.hand = []
    for _ in range(3):
      self.other_player.hand.append(self.deck.draw())

    self.briscola: Card = self.deck.draw()
    self.deck.cards.append(self.briscola)
    self.order = np.random.randint(0, 1)

    if self.order == 1:
      other_card = self.other_player.play_card(self.public_state())
      self.table.append(other_card)

  def _obs(self):
    return np.array(flatten(flatten(list(self.public_state().as_dict(played=False).values())))).squeeze()

  def _reset(self):
    self.init_state()
    self._episode_ended = False
    return ts.restart(self._obs())

  def _step(self, action):
    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    self.turn += 1
    while action >= len(self.my_player.hand):
        action = action - 1
    my_card = self.my_player.hand.pop(action)
    self.table.append(my_card)
    if self.order == 0:
        other_card = self.other_player.play_card(self.public_state())
        self.table.append(other_card)
    # i_winner: 0 -> Wins first player, 1 -> Wins second player
    i_winner = select_winner(self.table, self.briscola)
    reward = self._state_update_after_winner(i_winner)
    self._draw_phase()
    if self.order == 1 and not len(self.other_player.hand) == 0:
        other_card = self.other_player.play_card(self.public_state())
        self.table.append(other_card)

    # Make sure episodes don't go on forever.
    self._episode_ended = self.is_terminated()

    if self._episode_ended:
      return ts.termination(self._obs(), reward)
    else:
      return ts.transition(self._obs(), reward=reward, discount=1.0)

  def _state_update_after_winner(self, i_winner: int):
    """
    :param i_winner: 0 -> Wins first, 1 -> Wins second
    :return: The reward
    """
    i_winner = 0 if self.order == i_winner else 1
    reward = gained_points = sum(values_points[c.value] for c in self.table)
    self.points[i_winner] += gained_points
    self.my_played.append(self.table[self.order])
    self.other_played.append(self.table[1 - self.order])
    gained_points_my_player = gained_points_other_player = gained_points
    if i_winner == 0:
        self.my_points += gained_points
        self.order = 0
        gained_points_other_player = gained_points_other_player * -1
    else:
        self.other_points += gained_points
        self.order = 1
        gained_points_my_player = gained_points_my_player * -1
        reward = reward * -1
    self.my_player.notify_turn_winner(gained_points_my_player)
    self.other_player.notify_turn_winner(gained_points_other_player)

    penalty = self.get_penalty(reward)

    self.table = []
    sparse_reward = self.get_sparse_reward()

    dense_reward = reward if self.dense_reward else 0

    return dense_reward + sparse_reward - penalty

  def get_sparse_reward(self):
    if not self.sparse_reward:
        return 0

    sparse_reward = 0
    if self.points[0] > 60:
        sparse_reward = 60
    elif self.points[1] > 60:
        sparse_reward = -60
    return sparse_reward

  def get_penalty(self, reward):
    if not self.penalty:
        return 0

    penalty = -1000
    if self.order == 1:
        for card in self.my_player.hand:
            possible_table = [self.table[0], card]
            winner = select_winner(possible_table, self.briscola)
            coef_pts = 1 if winner else -1
            gain = coef_pts * sum(map(lambda c: c.points, possible_table))
            if gain > reward and gain > penalty and card.suit != self.briscola.suit:
                penalty = gain
    if penalty == -1000:
        penalty = 0
    elif penalty < 0:
        penalty = abs(reward - penalty)
    return penalty

  def _draw_phase(self):
    if not self.deck.is_empty():
      c1 = self.deck.draw()
      c2 = self.deck.draw()
      if self.order == 0:
          c_my_player = c1
          c_other_player = c2
      else:
          c_other_player = c1
          c_my_player = c2
      self.my_player.hand.append(c_my_player)
      self.other_player.hand.append(c_other_player)

  def is_terminated(self):
    return any(p > 60 for p in self.points) or \
        (self.deck.is_empty() and all(len(p.hand) == 0 for p in self.players))

  def public_state(self):
      return PublicState(self.my_points, self.other_points, self.my_player.hand,
                         len(self.other_player.hand), len(self.deck.cards),
                         self.table, [], [],
                         self.turn, self.briscola, self.order)


def compute_avg_return(environment, policy, num_episodes=10):
  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


def collect_episode(environment, policy, num_episodes, rb_observer):
  driver = py_driver.PyDriver(
    environment,
    py_tf_eager_policy.PyTFEagerPolicy(
      policy, use_tf_function=True),
    [rb_observer],
    max_episodes=num_episodes)
  initial_time_step = environment.reset()
  driver.run(initial_time_step)


def train_reinforce(
        num_iterations: int = 10000,
        collect_episodes_per_iteration: int = 5,
        replay_buffer_capacity: int = 2000,
        fc_layer_params = (100,),
        learning_rate = 1e-3,
        log_interval = 100,
        num_eval_episodes = 100,
        eval_interval = 500,
        optimizer = 'adam',
        penalty=False,
        sparse_reward=False,
        dense_reward=False
):

    train_py_env = BriscolaTFEnv(sparse_reward=sparse_reward, dense_reward=dense_reward, penalty=penalty)
    eval_py_env = BriscolaTFEnv(sparse_reward=sparse_reward, dense_reward=dense_reward, penalty=penalty)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        print("Undefined optimizer "+optimizer)
        return

    train_step_counter = tf.Variable(0)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)
    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter
    )

    tf_agent.initialize()

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        tf_agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        table_name=table_name,
        sequence_length=None,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddEpisodeObserver(
        replay_buffer.py_client,
        table_name,
        replay_buffer_capacity
    )

    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

      # Collect a few episodes using collect_policy and save to the replay buffer.
      collect_episode(train_py_env, tf_agent.collect_policy, collect_episodes_per_iteration, rb_observer)

      # Use data from the buffer and update the agent's network.
      iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
      trajectories, _ = next(iterator)
      train_loss = tf_agent.train(experience=trajectories)

      replay_buffer.clear()

      step = tf_agent.train_step_counter.numpy()

      if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

      if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)


def train_dqn(
        num_iterations: int = 10000,
        replay_buffer_max_length: int = 100000,
        fc_layer_params=(100,),
        log_interval=100,
        num_eval_episodes=100,
        eval_interval=500,
        batch_size=64,
        collect_steps_per_iteration=1,
        optimizer='adam',
        penalty=False,
        sparse_reward=False,
        dense_reward=False,
        learning_rate=1e-4
):
    train_step_counter = tf.Variable(0)

    train_py_env = BriscolaTFEnv(sparse_reward=sparse_reward, dense_reward=dense_reward, penalty=penalty)
    eval_py_env = BriscolaTFEnv(sparse_reward=sparse_reward, dense_reward=dense_reward, penalty=penalty)

    #train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    #eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    train_env = train_py_env
    eval_env = eval_py_env

    action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        print("Undefined optimizer "+optimizer)
        return

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        td_errors_loss_fn=common.element_wise_squared_loss,
        optimizer=optimizer,
        train_step_counter=train_step_counter
    )
    agent.initialize()
    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        table_name,
        sequence_length=2)

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    # Reset the environment.
    time_step = train_py_env.reset()

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        train_env,
        py_tf_eager_policy.PyTFEagerPolicy(
            agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    for _ in range(num_iterations):

        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)


def train_sac(
        num_iterations = 100000,
        initial_collect_steps = 10000,
        batch_size = 256,
        critic_learning_rate = 3e-4,
        actor_learning_rate = 3e-4,
        alpha_learning_rate = 3e-4,
        target_update_tau = 0.005,
        target_update_period = 1,
        gamma = 0.99,
        reward_scale_factor = 1.0,
        actor_fc_layer_params = (256, 256),
        critic_joint_fc_layer_params = (256, 256),
        replay_buffer_capacity = 10000,
        log_interval = 5000,
        num_eval_episodes = 20,
        eval_interval = 10000,
        policy_save_interval = 5000,
        penalty=False,
        sparse_reward=False,
        dense_reward=False
):
    tempdir = tempfile.gettempdir()

    train_py_env = BriscolaTFEnv(sparse_reward=sparse_reward, dense_reward=dense_reward, penalty=penalty)
    eval_py_env = BriscolaTFEnv(sparse_reward=sparse_reward, dense_reward=dense_reward, penalty=penalty)

    train_env = train_py_env
    eval_env = eval_py_env

    use_gpu = True

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(train_env))

    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')
    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))
    with strategy.scope():
        train_step = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.keras.optimizers.Adam(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.keras.optimizers.Adam(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            train_step_counter=train_step)

        tf_agent.initialize()

    rate_limiter = reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3,
                                                            error_buffer=3.0)
    table_name = 'uniform_table'
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))

    reverb_server = reverb.Server([table])

    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        sequence_length=2,
        table_name=table_name,
        local_server=reverb_server)

    dataset = reverb_replay.as_dataset(
        sample_batch_size=batch_size, num_steps=2).prefetch(50)
    experience_dataset_fn = lambda: dataset

    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_eval_policy, use_tf_function=True)

    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True)

    random_policy = random_py_policy.RandomPyPolicy(
        train_env.time_step_spec(), train_env.action_spec())

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client,
        table_name,
        sequence_length=2,
        stride_length=1)

    initial_collect_actor = actor.Actor(
        train_env,
        random_policy,
        train_step,
        steps_per_run=initial_collect_steps,
        observers=[rb_observer])
    initial_collect_actor.run()

    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
        train_env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
        observers=[rb_observer, env_step_metric])

    eval_actor = actor.Actor(
        eval_env,
        eval_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        summary_dir=os.path.join(tempdir, 'eval'),
    )

    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=policy_save_interval),
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    agent_learner = learner.Learner(
        tempdir,
        train_step,
        tf_agent,
        experience_dataset_fn,
        triggers=learning_triggers,
        strategy=strategy)

    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
        return results

    metrics = get_eval_metrics()

    def log_eval_metrics(step, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))

    log_eval_metrics(0, metrics)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]

    for _ in range(num_iterations):
        # Training.
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy

        if eval_interval and step % eval_interval == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])

        if log_interval and step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

    rb_observer.close()
    reverb_server.stop()

if __name__ == '__main__':
    #train_dqn(
    #    num_iterations=10000,
    #    replay_buffer_max_length=100_000,
    #    fc_layer_params=(100,50),
    #    log_interval=100,
    #    num_eval_episodes=100,
    #    eval_interval=500,
    #    batch_size=64,
    #    collect_steps_per_iteration=1,
    #    penalty=True,
    #    dense_reward=True,
    #    sparse_reward=True
    #)
    #train_sac(
    #    penalty=True,
    #    dense_reward=True,
    #    sparse_reward=True
    #)
    train_reinforce(
        num_iterations=10000,
        log_interval=100,
        eval_interval=500,
        num_eval_episodes=100,
        learning_rate=1e-4,
        fc_layer_params=(100,50),
        replay_buffer_capacity=4000,
        penalty = True,
        dense_reward = True,
        sparse_reward = True
    )