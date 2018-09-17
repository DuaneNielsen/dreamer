import torch
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import matplotlib.pyplot as plt
from models import MDNRNN
from mentalitystorm import Storeable, config, Demo, MseKldLoss, OpenCV, Observable, dispatcher
import data
import torchvision
import torchvision.transforms as TVT


def max_seq_length(observation_minibatch):
    max_length = 0
    for observations in observation_minibatch:
        seq_len = observations.size(0)
        if seq_len > max_length:
            max_length = seq_len
    return max_length


def timeline(batch_size, timesteps, z_size, device):
    timeline = torch.linspace(0, timesteps - 1, timesteps).unsqueeze(1).repeat(1, z_size).to(device)
    batch = []
    for _ in range(batch_size):
        batch.append(timeline)
    return batch

if __name__ == '__main__':

    batch_size = 10
    z_size = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dispatcher.registerView('output', OpenCV('output', (420, 320)))

    dataset = data.ActionEncoderDataset(config.datapath('SpaceInvaders-v4/latent'))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data.collate_action_observation, drop_last=True)

    convolutions = Storeable.load('GM53H301W5YS38XH')
    convolutions.decoder.register_forward_hook(Observable.send_output_as_image)

    model = MDNRNN(z_size, 256, 1, 10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        loss = None
        for observation_minibatch, action_minibatch in loader:
            seq_length = max_seq_length(observation_minibatch)
            tl = timeline(batch_size, seq_length, z_size, device)
            pi, mu, sigma, _ = model(tl)
            padded_obs = rnn_utils.pad_sequence(observation_minibatch, batch_first=True).squeeze().to(device)
            loss = model.loss_fn(padded_obs, pi, mu, sigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Loss: ' + str(loss.item()))