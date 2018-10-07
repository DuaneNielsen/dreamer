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


""" Changes observation space to vector series in observation space
"""
