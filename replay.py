from mentalitystorm.data import ActionEncoderDataset, collate_action_observation
from mentalitystorm.config import config
from mentalitystorm.observe import UniImageViewer
import torch.utils.data as data_utils
from tqdm import tqdm


if __name__ == '__main__':

    uni_viewer = UniImageViewer('d1', (320, 480))
    dataset = ActionEncoderDataset(config.datapath('SpaceInvaders-v4/rl_raw_v2'))
    loader = data_utils.DataLoader(dataset, batch_size=4, collate_fn=collate_action_observation)
    loader = tqdm(loader, desc='description')

    for screen, observation, action, reward, done, latent in loader:
        for frame in screen[0]:
            uni_viewer.render(frame)
