import torch
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as data_utils

from models import MDNRNN
from mentalitystorm.storage import Storeable
from mentalitystorm.config import config
from mentalitystorm.observe import UniImageViewer
from mentalitystorm.data import ActionEncoderDataset, collate_action_observation

from tensorboardX import SummaryWriter
from tqdm import tqdm

from statistics import mean

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

    batch_size = 128
    i_size = 16 + 6
    z_size = 16
    train_model = True
    config.increment('run_id')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ActionEncoderDataset(config.datapath('SpaceInvaders-v4/rl_raw_v2'), load_observation=False, load_screen=False)

    dev = data_utils.Subset(dataset, range(dataset.count * 2 // 10))
    train = data_utils.Subset(dataset, range(0, dataset.count * 9//10))
    test = data_utils.Subset(dataset, range(dataset.count * 9 // 10 + 1, dataset.count))

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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def temporal_shift_and_diff(action, latent):
        action_latent = [torch.cat(act_z, dim=1)[:-1] for act_z in zip(action, latent)]
        z_plus_1_delta = []
        for z in latent:
            z_plus_1 = z.clone()[1:]
            z_plus_1_delta.append(z_plus_1 - z[:-1])
        return action_latent, z_plus_1_delta


    for epoch in range(1, 21):
        train = tqdm(train, desc=f'train {epoch}')
        losses = []
        for screen, observation, action, reward, done, latent in train:

            #seq_length = max_seq_length(observation_minibatch)
            #tl = timeline(batch_size, seq_length, z_size, device)

            action_latent = [torch.cat(act_z, dim=1)[:-1] for act_z in zip(action, latent)]
            latent_t_plus_1 = [t.clone()[1:] for t in latent]
            pi, mu, sigma, _ = model(action_latent, device)
            padded_latent_t_plus_1 = rnn_utils.pad_sequence(latent_t_plus_1, batch_first=True).squeeze().to(device)
            #padded_latent = padded_latent.clamp(1e-1, 1e1)
            loss = model.loss_fn(padded_latent_t_plus_1, pi, mu, sigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            train.set_description(f'train epoch: {epoch} loss : {mean(losses)}')
            tb.add_scalar('train/loss', loss.item(), global_step)
            tb.add_scalar('train/sigma', sigma.mean().item(), global_step)
            global_step += 1

        test = tqdm(test, desc=f'test {epoch}')
        losses = []
        for screen, observation, action, reward, done, latent in test:
            action_latent = [torch.cat(act_z, dim=1)[:-1] for act_z in zip(action, latent)]
            latent_t_plus_1 = [t.clone()[1:] for t in latent]
            pi, mu, sigma, _ = model(action_latent, device)
            padded_latent_t_plus_1 = rnn_utils.pad_sequence(latent_t_plus_1, batch_first=True).squeeze().to(device)
            loss = model.loss_fn(padded_latent_t_plus_1, pi, mu, sigma)
            losses.append(loss.item())
            test.set_description(f'test epoch : {epoch} loss : {mean(losses)}')
            tb.add_scalar('test/loss', loss.item(), global_step)
            tb.add_scalar('test/sigma', sigma.mean().item(), global_step)
            y_pred = model.sample(pi, mu, sigma)
            y_pred = y_pred[0].squeeze()
            global_step += 1

        if epoch % 1 == 0:
            model.save(config.model_fn(model))

