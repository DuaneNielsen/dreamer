import torch
import torch.utils.data
from mentalitystorm import ImageViewer

image_viewer = ImageViewer('action', (320, 480))


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
    # first frame has ridiculous high variance, so drop it, I
    #clean = drop_first_frame(minibatch)
    #delta = observation_deltas(clean)
    return minibatch


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
