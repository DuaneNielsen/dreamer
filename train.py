import torch
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt

import mentalitystorm.atari
from models import MDNRNN
from mentalitystorm import Storeable, config, Demo, MseKldLoss, OpenCV, Observable, dispatcher, ImageChannel
import data
from tensorboardX import SummaryWriter
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

    batch_size = 140
    z_size = 16
    train_model = False
    config.increment('run_id')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dispatcher.registerView('output', OpenCV('output', (320, 480)))
    dispatcher.registerView('ground_truth', OpenCV('ground_truth', (320, 480)))

    dataset = mentalitystorm.atari.ActionEncoderDataset(config.datapath('SpaceInvaders-v4/latent'))
    dev = data_utils.Subset(dataset, range(dataset.count * 2 // 10))
    train = data_utils.Subset(dataset, range(0, dataset.count * 9//10))
    test = data_utils.Subset(dataset, range(dataset.count * 9 // 10 + 1, dataset.count))
    dev = data_utils.DataLoader(dev, batch_size=batch_size, collate_fn=data.collate_action_observation,
                                 drop_last=True, )
    train = data_utils.DataLoader(train, batch_size=batch_size, collate_fn=data.collate_action_observation, drop_last=True,)
    test = data_utils.DataLoader(test, batch_size=batch_size, collate_fn=data.collate_action_observation,
                                  drop_last=True, )

    convolutions = Storeable.load('C:\data\models\GM53H301W5YS38XH').to(device)
    convolutions.decoder.register_forward_hook(Observable.send_output_as_image)

    ground_truth_decoder = Storeable.load('C:\data\models\GM53H301W5YS38XH').to(device)
    ground_truth_decoder.decoder.register_forward_hook(ImageChannel('ground_truth').send_output_as_image)

    #model = MDNRNN(i_size=6, z_size=z_size, hidden_size=256, num_layers=1, n_gaussians=5).to(device)
    # model trained directly on latent space
    model = Storeable.load(r'C:\data\runs\421\mdnrnn-i_size-6-z_size-16-hidden_size-256-num_layers-1-n_gaussians-5_4.md').to(device)

    tb = SummaryWriter(config.tb_run_dir(model))
    global_step = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        loss = None
        for raw_latent, first_frame, delta_latent, action_minibatch in train:

            #seq_length = max_seq_length(observation_minibatch)
            #tl = timeline(batch_size, seq_length, z_size, device)
            if train_model:
                pi, mu, sigma, _ = model(action_minibatch, device)
                padded_obs = rnn_utils.pad_sequence(delta_latent, batch_first=True).squeeze().to(device)
                padded_obs = padded_obs.clamp(1e-1, 1e1)
                loss = model.loss_fn(padded_obs, pi, mu, sigma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tb.add_scalar('train/loss', loss.item(), global_step)
                tb.add_scalar('train/sigma', sigma.mean().item(), global_step)
                global_step += 1

        if epoch % 1 == 0:
            for raw_latent, first_frame, delta_latent, action_minibatch in test:
                #tl = timeline(1, 20, z_size, device)
                pi, mu, sigma, _ = model(action_minibatch, device)
                padded_obs = rnn_utils.pad_sequence(delta_latent, batch_first=True).squeeze().to(device)
                loss = model.loss_fn(padded_obs, pi, mu, sigma)
                print('Loss: ' + str(loss.item()))
                tb.add_scalar('test/loss', loss.item(), global_step)
                tb.add_scalar('test/sigma', sigma.mean().item(), global_step)
                y_pred = model.sample(pi, mu, sigma)
                y_pred = y_pred[0].squeeze()
                global_step += 1

                ds = data.DeltaStream(first_frame[0])

                for i, delta in enumerate(delta_latent[0]):
                    from_delta = ds.delta_to_frame(delta_latent[0][i, :, :]).to(device)
                    original = raw_latent[0][i, :, :].to(device)
                    convolutions.decoder(from_delta.unsqueeze(0))
                    ground_truth_decoder.decoder(original.unsqueeze(0))
            model.save(config.model_fn(model))

