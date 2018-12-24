from mentalitystorm.instrumentation import TB, register_tb, LatentInstrument, GUIProgressMeter, UniImageViewer
from mentalitystorm.data import StandardSelect, GymImageDataset, DataPackage

from models.train import SimpleRunFac, Params, Run
from models.losses import MseKldLoss
from models.util import Handles
from models.config import config

import torchvision.transforms as TVT
from models.cnn import ConvVAE4Fixed
from torch.optim import Adam

if __name__ == '__main__':

    input_viewer = UniImageViewer('input', (320, 480))
    output_viewer = UniImageViewer('output', (320, 480))
    latent_viewer = UniImageViewer('latent', (320, 480))
    latent_instr = LatentInstrument()

    co_ord_conv_shots = GymImageDataset(directory=config.datapath(r'SpaceInvaders-v4\images\raw_v1\all'),
                                        input_transform=TVT.Compose([TVT.ToTensor()]),
                                        target_transform=TVT.Compose([TVT.ToTensor()]))

    co_ord_conv_data_package = DataPackage(co_ord_conv_shots, StandardSelect(target_index=0))

    run_fac = SimpleRunFac()
    opt = Params(Adam, lr=1e-3)
    run_fac.run_list.append(Run(Params(ConvVAE4Fixed, (210, 160), 64, variational=True),
                                opt,
                                Params(MseKldLoss),
                                co_ord_conv_data_package,
                                run_name='full_v1'))

    #run_fac = SimpleRunFac.resume(r'C:\data\runs\549', co_ord_conv_data_package)
    batch_size = 64
    epochs = 30

    for model, opt, loss_fn, data_package, trainer, tester, run in run_fac:
        dev, train, test, selector = data_package.loaders(batch_size=batch_size)

        model.register_forward_hook(input_viewer.view_input)
        model.decoder.register_forward_hook(latent_viewer.view_input)
        model.register_forward_hook(output_viewer.view_output)
        register_tb(run, config)
        gui = GUIProgressMeter(description='training')
        trainer.register_after_hook(gui.update_train)
        tester.register_after_hook(gui.update_test)

        for epoch in run.for_epochs(epochs):

            #epoch.register_after_hook(write_histogram)
            epoch.register_after_hook(gui.end_epoch)
            epoch.execute_before(epoch)

            handles = Handles()

            trainer.train(model, opt, loss_fn, dev, selector, run, epoch)

            handles += loss_fn.register_hook(TB().tb_test_loss_term)
            tester.test(model, loss_fn, dev, selector, run, epoch)

            epoch.execute_after(epoch)
            handles.remove()
            run.save()
