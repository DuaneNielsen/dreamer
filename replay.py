import torch
import data
from mentalitystorm import config, Observable, Storeable, OpenCV, dispatcher, ImageChannel

if __name__ == '__main__':

    dataset = data.ActionEncoderDataset(config.datapath('SpaceInvaders-v4/latent'))

    loader = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=data.collate_action_observation)

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
        #assert raw_latent[0][0, 0, 0] - first_frame[0][0, 0, 0] == delta_latent[0][0, 0, 0]

        ds = data.DeltaStream(first_frame[0])
        #zero_delta = torch.zeros(first_frame[0].shape)
        #assert ds.delta_to_frame(zero_delta).sum() == first_frame[0].sum()

        for i, delta in enumerate(delta_latent[0]):

            from_delta = ds.delta_to_frame(delta_latent[0][i, :, :])
            original = raw_latent[0][i, :, :]
            d1.decoder(original.unsqueeze(0))
            d2.decoder(from_delta.unsqueeze(0))




            # first_frame = None
            # running_delta = torch.zeros(1, item.shape[1], 1, 1)
            # prev_frame = None
            # for i, frame in enumerate(item):
            #     if i <= 0:
            #         first_frame = frame.unsqueeze(0)
            #         prev_frame = frame.unsqueeze(0)
            #         convolutions.decoder(frame.unsqueeze(0))
            #     else:
            #         delta = frame - prev_frame
            #         running_delta = running_delta + delta
            #         theframe = first_frame + running_delta
            #         decoder2.decoder(theframe)
            #         convolutions.decoder(frame.unsqueeze(0))
            #         prev_frame = frame


