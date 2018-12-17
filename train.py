import torch
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as data_utils

from models import MDNRNN
from data import RandomSubSequence

from mentalitystorm.storage import Storeable
from mentalitystorm.config import config
from mentalitystorm.observe import UniImageViewer
from mentalitystorm.data import ActionEncoderDataset, collate_action_observation

from tensorboardX import SummaryWriter
from tqdm import tqdm
from statistics import mean
from pathlib import Path

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


class LossRecorder:
    def __init__(self, description, tb=None, tqdm=None):
        self.epoch = 0
        self.losses = []
        self.description = description
        self.tb = tb
        self.tqdm = tqdm

    def record(self, epoch, global_step, loss, tqdm=None):
        if epoch != self.epoch:
            self.losses = []
        self.losses.append(loss.item())
        if tqdm is not None:
            tqdm.set_description(f'{self.description} epoch: {epoch} loss : {mean(self.losses)}')
        if self.tb is not None:
            self.tb.add_scalar(f'{self.description}/loss', loss.item(), global_step)
            self.tb.add_scalar(f'{self.description}/sigma', sigma.mean().item(), global_step)

    def loss(self):
        return mean(self.losses)

if __name__ == '__main__':

    batch_size = 128
    i_size = 16 + 6
    z_size = 16
    train_model = True
    config.increment('run_id')
    clamp = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ActionEncoderDataset(config.datapath('SpaceInvaders-v4/rl_raw_v2'), load_observation=False, load_screen=False)
    dataset = RandomSubSequence(dataset, 100)

    dev = data_utils.Subset(dataset, range(len(dataset) * 2 // 10))
    train = data_utils.Subset(dataset, range(0, len(dataset) * 9//10))
    test = data_utils.Subset(dataset, range(len(dataset) * 9 // 10 + 1, len(dataset)))

    dev = data_utils.DataLoader(dev, batch_size=batch_size, collate_fn=collate_action_observation,
                                drop_last=False, )
    train = data_utils.DataLoader(train, batch_size=batch_size, collate_fn=collate_action_observation, drop_last=False, )
    test = data_utils.DataLoader(test, batch_size=batch_size, collate_fn=collate_action_observation,
                                 drop_last=False, )

    convolutions = Storeable.load('C:\data\models\GM53H301W5YS38XH').to(device)
    convolutions.decoder.register_forward_hook(UniImageViewer('decoded').view_output)

    ground_truth_decoder = Storeable.load('C:\data\models\GM53H301W5YS38XH').to(device)
    ground_truth_decoder.decoder.register_forward_hook(UniImageViewer('ground_truth').view_input)

    model = MDNRNN(i_size=i_size, z_size=z_size, hidden_size=32, num_layers=3, n_gaussians=3).to(device)

    #model = Storeable.load(r'C:\data\runs\687\mdnrnn-i_size-6-z_size-16-hidden_size-256-num_layers-1-n_gaussians-5_10.md').to(device)

    tb = SummaryWriter(config.tb_run_dir(model))
    global_step = 0
    train_lr = LossRecorder('train', tb)
    test_lr = LossRecorder('test', tb)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def temporal_shift_and_diff(latent):
        """
        computes the delta between timesteps
        :param latent: a list of observations
        :return: a list of the differences, shortened by 1
        """
        z_plus_1_delta = []
        for z in latent:
            z_plus_1 = z.clone()[1:]
            z_plus_1_delta.append(z_plus_1 - z[:-1])
        return z_plus_1_delta


    for epoch in range(1, 101):
        train = tqdm(train, desc=f'train {epoch}')
        for screen, observation, action, reward, done, latent in train:

            action_latent = [torch.cat(act_z, dim=1)[:-1] for act_z in zip(action, latent)]
            latent_t_plus_1 = temporal_shift_and_diff(latent)
            pi, mu, sigma, _ = model(action_latent, device)
            padded_latent_t_plus_1 = rnn_utils.pad_sequence(latent_t_plus_1, batch_first=True).squeeze().to(device)
            if clamp:
                padded_latent = padded_latent_t_plus_1.clamp(1e-1, 1e1)
            loss = model.loss_fn(padded_latent_t_plus_1, pi, mu, sigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_lr.record(epoch, global_step, loss, train)
            global_step += 1

        test = tqdm(test, desc=f'test {epoch}')
        for screen, observation, action, reward, done, latent in test:
            action_latent = [torch.cat(act_z, dim=1)[:-1] for act_z in zip(action, latent)]
            latent_t_plus_1 = temporal_shift_and_diff(latent)
            pi, mu, sigma, _ = model(action_latent, device)
            padded_latent_t_plus_1 = rnn_utils.pad_sequence(latent_t_plus_1, batch_first=True).squeeze().to(device)
            loss = model.loss_fn(padded_latent_t_plus_1, pi, mu, sigma)
            y_pred = model.sample(pi, mu, sigma)
            y_pred = y_pred[0].squeeze()
            test_lr.record(epoch, global_step, loss, test)
            global_step += 1

        if epoch % 1 == 0:
            model_file = Path(f'runs/1/model_epoch_{epoch}_loss_{test_lr.loss():.4f}.wgt')
            model_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(model_file))

