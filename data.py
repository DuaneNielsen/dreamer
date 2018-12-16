import torch
import torch.utils.data as dt_util
from random import randrange


class DeltaStream:
    def __init__(self, first_frame):
        self.first_frame = first_frame
        self.running_delta = torch.zeros(first_frame.shape[0], 1, 1)

    def delta_to_frame(self, delta):
        self.running_delta = self.running_delta + delta
        return self.first_frame + self.running_delta


""" Changes observation space to vector series in observation space
"""


class RandomSubSequence(dt_util.Dataset):
    def __init__(self, action_encoder_dataset, subsequence_length):
        self.dataset = action_encoder_dataset
        self.length = subsequence_length

    def __getitem__(self, item):
        """
        Returns a random sub-sequence from the episode
        """
        screen, obs, action, reward, done, latent = self.dataset[item]
        start = randrange(len(action) - self.length)
        end = start + self.length
        if screen is not None:
            screen = screen[start:end]
        obs = obs[start:end]
        action = action[start:end]
        reward = reward[start:end]
        done = done[start:end]
        latent = latent[start:end]
        return screen, obs, action, reward, done, latent

    def __len__(self):
        return len(self.dataset)