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


def collate_action_observation(batch):
        return [list(t)for t in zip(*batch)]