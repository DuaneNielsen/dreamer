from unittest import TestCase

import mentalitystorm.atari
from mentalitystorm import config
import data
import torch
import torch.utils.data as data_utils



class TestData(TestCase):
    def test_data(self):
        dataset = mentalitystorm.atari.ActionEncoderDataset(config.datapath('SpaceInvaders-v4/latent'))
        devset = data_utils.Subset(dataset, range(1))
        dev = data_utils.DataLoader(devset, batch_size=1, collate_fn=data.collate_action_observation)

        for raw_latent, first_frame, delta_latent, action_minibatch in dev:
            assert raw_latent[0][0, 0, 0] - first_frame[0][0, 0, 0] == delta_latent[0][0, 0, 0]

            ds = data.DeltaStream(first_frame[0])
            zero_delta = torch.zeros(first_frame[0].shape)
            assert ds.delta_to_frame(zero_delta).sum() == first_frame[0].sum()

            for i, delta in enumerate(delta_latent[0]):

                from_delta = ds.delta_to_frame(delta_latent[0][i, :, :])[0, 0, 0].item()
                original = raw_latent[0][i, 0, 0].item()
                print('test ' + str(i))
                if from_delta != original:
                    print('delta %.15f' % delta_latent[0][i, 0, 0].item())
                    print('frame from delta %.15f' % from_delta)
                    print('original %.15f' % original)
                import math
                assert math.isclose(from_delta, original, rel_tol=1e-09, abs_tol=1e-05)


