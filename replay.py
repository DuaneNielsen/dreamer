import torch
import data
import mentalitystorm.atari
from mentalitystorm import config, Observable, Storeable, OpenCV, dispatcher, ImageChannel

if __name__ == '__main__':

    dataset = mentalitystorm.atari.ActionEncoderDataset(config.datapath('SpaceInvaders-v4/latent'))

    loader = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=mentalitystorm.atari.collate_action_observation)

    d1 = Storeable.load('C:\data\models\GM53H301W5YS38XH')
    d2 = Storeable.load('C:\data\models\GM53H301W5YS38XH')

    def send_output(self, input, output):
        obs = Observable()
        obs.updateObserversWithImage('output', output.data, 'tensorPIL')

    d1.decoder.register_forward_hook(Observable.send_output_as_image)
    d2.decoder.register_forward_hook(ImageChannel('delta').send_output_as_image)

    dispatcher.registerView('output', OpenCV('output', (420, 320)))
    dispatcher.registerView('delta', OpenCV('delta', (420, 320)))

    for raw_latent, first_frame, delta_latent, action_minibatch in loader:

        ds = data.DeltaStream(first_frame[0])

        for i, delta in enumerate(delta_latent[0]):

            from_delta = ds.delta_to_frame(delta_latent[0][i, :, :])
            original = raw_latent[0][i, :, :]
            d1.decoder(original.unsqueeze(0))
            d2.decoder(from_delta.unsqueeze(0))
