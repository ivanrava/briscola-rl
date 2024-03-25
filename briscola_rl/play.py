from stable_baselines3 import DQN

from game import BriscolaInteractivePlayer
import torch

if __name__ == '__main__':
    env = BriscolaInteractivePlayer()
    model = DQN.load('../checkpoints/true-star')

    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, _, info = env.step(action)
        if terminated:
            break

