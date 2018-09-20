import torch
import torch.utils.data
from pathlib import Path


class ActionEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        torch.utils.data.Dataset.__init__(self)
        self.path = Path(directory)
        self.count = 0
        for _ in self.path.iterdir():
            self.count += 1

    def __getitem__(self, index):
        filepath = self.path / str(index)
        import pickle
        with open(filepath.absolute(), 'rb') as f:
            oa = pickle.load(f)
        return torch.Tensor(oa.observation), torch.Tensor(oa.action)

    def __len__(self):
        return self.count


class DeltaStream:
    def __init__(self, first_frame):
        self.first_frame = first_frame
        self.running_delta = torch.zeros(first_frame.shape[0], 1, 1)

    def delta_to_frame(self, delta):
        self.running_delta = self.running_delta + delta
        return self.first_frame + self.running_delta


def collate_action_observation(batch):
    # short longest to shortest
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    minibatch = [list(t) for t in zip(*batch)]
    # first frame has ridiculous high variance, so drop it
    clean = drop_first_frame(minibatch)
    delta = observation_deltas(clean)
    return delta


def drop_first_frame(minibatch):
    latent_l = []
    action_l = []
    for latent in minibatch[0]:
        latent_l.append(latent[1:, :])
    for action in minibatch[1]:
        action_l.append(action[1:, :])
    return latent_l, action_l


""" Changes observation space to vector series in observation space
"""
def observation_deltas(minibatch):
    latent_l = []
    latent_raw_l = []
    action_l = []
    first_frame_l = []

    for first_frame in minibatch[0]:
        first_frame_l.append(first_frame[0, :])

    for latent in minibatch[0]:
        latent_l.append(latent[1:, :] - latent[:-1, :])

    for action in minibatch[1]:
        action_l.append(action[:-1, :])

    for latent in minibatch[0]:
        latent_raw_l.append(latent[1:, :])

    return latent_raw_l, first_frame_l, latent_l, action_l
