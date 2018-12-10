import torch
from mentalitystorm.storage import Storeable
from mentalitystorm.config import config
from mentalitystorm.data_containers import ActionEmbedding
import gym

gym_environment = 'SpaceInvaders-v4'

device = config.device()
model = Storeable.load(r'C:\data\runs\687\mdnrnn-i_size-6-z_size-16-hidden_size-256-num_layers-1-n_gaussians-5_2.md').to(device)
controller = torch.load(r'.\modelzoo\best_model68').to(device)

env = gym.make(gym_environment)
start_action = ActionEmbedding(env=env).start_tensor().to(device)
context = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))

pi, mu, sigma, context = model.step(start_action.unsqueeze(0).unsqueeze(0), context)
observation = model.sample(pi, mu, sigma)

for frame in range(1000):
    action = controller(observation.double())
    _, index = action.max(2)
    action = torch.zeros_like(action)
    action[0, 0, index] = 1.0
    pi, mu, sigma, context = model.step(action.float(), context)
    observation = model.sample(pi, mu, sigma)
