from stable_baselines3 import DQN

from game import BriscolaInteractivePlayer
import torch

if __name__ == '__main__':
    env = BriscolaInteractivePlayer()
    model = DQN.load('./checkpoints/earnest-night')

    obs, _ = env.reset()
    while True:
        q_values = model.policy.q_net.forward(torch.from_numpy(obs))
        print(q_values)
        action, _states = model.predict(obs)
        obs, reward, terminated, _, info = env.step(action)
        if terminated:
            break

