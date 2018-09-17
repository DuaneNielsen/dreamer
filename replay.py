import torch
import data
from mentalitystorm import config, Observable, Storeable, OpenCV, dispatcher

if __name__ == '__main__':

    dataset = data.ActionEncoderDataset(config.datapath('SpaceInvaders-v4/latent'))

    loader = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=data.collate_action_observation)

    convolutions = Storeable.load('GM53H301W5YS38XH')

    def send_output(self, input, output):
        obs = Observable()
        obs.updateObserversWithImage('output', output.data, 'tensorPIL')

    convolutions.decoder.register_forward_hook(send_output)

    dispatcher.registerView('output', OpenCV('output', (420, 320)))

    for observation_minibatch, action_minibatch in loader:
        for item in observation_minibatch:
            for frame in item:
                theframe = frame.unsqueeze(0)
                convolutions.decoder(theframe)