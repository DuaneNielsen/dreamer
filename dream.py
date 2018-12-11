import torch
from mentalitystorm.storage import Storeable
from mentalitystorm.config import config
from mentalitystorm.data_containers import ActionEmbedding
import gym
from mentalitystorm.observe import UniImageViewer
from time import sleep


def main(model, controller, gym_environment, device, decoder=None):

    if decoder is not None:
        decoder = decoder.to(device)
        decoder.register_forward_hook(UniImageViewer('decoded_latent_space', (320, 480)).view_output)

    env = gym.make(gym_environment)
    start_action = ActionEmbedding(env=env).start_tensor().to(device)
    context = (torch.zeros(1, 1, 32).to(device), torch.zeros(1, 1, 32).to(device))
    pi, mu, sigma, context = model.step(start_action.unsqueeze(0).unsqueeze(0))
    observation = model.sample(pi, mu, sigma)

    for frame in range(1000):
        if decoder is not None:
            decoder(observation.unsqueeze(0).transpose(3,1))
        action = controller(observation.double())
        _, index = action.max(2)
        action = torch.zeros_like(action)
        action[0, 0, index] = 1.0
        pi, mu, sigma, context = model.step(action.float(), context)
        observation = model.sample(pi, mu, sigma)
        sleep(0.05)


if __name__ == '__main__':

    gym_environment = 'SpaceInvaders-v4'
    device = config.device()
    model = Storeable.load(
        r'C:\data\runs\698\mdnrnn-i_size-6-z_size-16-hidden_size-32-num_layers-5-n_gaussians-3_3.md').to(device)
    controller = torch.load(r'.\modelzoo\best_model68').to(device)

    visuals = Storeable.load('.\modelzoo\GM53H301W5YS38XH')
    decoder = visuals.decoder

    main(model, controller, gym_environment, device, decoder)
