import torch
import numpy as np
import matplotlib.pyplot as plt
from models import MDNRNN
from mentalitystorm import Storeable, config, Demo, MseKldLoss, OpenCV
import torchvision
import torchvision.transforms as TVT

if __name__ == '__main__':

    dataset = torchvision.datasets.ImageFolder(
        root=config.datapath('spaceinvaders/images/raw'),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    convolutions = Storeable.load('GM53H301W5YS38XH')

    demo = Demo()
    convolutions.registerView('z_corr', OpenCV('z_corr', (512, 512)))
    # lossfunc = MseKldLoss()
    # demo.test(convolutions, dataset, 128, lossfunc)
    demo.rotate(convolutions, 16)
    demo.sample(convolutions, 16, samples=20)
    demo.demo(convolutions, dataset)

    n_samples = 500
    epsilon = torch.randn(n_samples)
    x_data = torch.linspace(0, 50, n_samples)
    y_data = 7 * np.sin(0.75 * x_data) + 0.5 * x_data + epsilon

    y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    line1 = ax.scatter(y_data.numpy(), x_data.numpy())
    line2 = None
    fig.canvas.draw()

    model = MDNRNN(1, 256, 2, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x_data = x_data.unsqueeze(0).to(device)
    y_data = y_data.unsqueeze(0).to(device)

    for epoch in range(10000):
        pi, mu, sigma, _ = model(x_data)
        loss = model.loss_fn(y_data, pi, mu, sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('loss: ' + str(loss.item()))
            pi, mu, sigma, _ = model(x_data)

            y_pred = model.sample(pi, mu, sigma)

            x_plot = x_data.data.squeeze().cpu().numpy()
            y_plot = y_pred.data.squeeze().cpu().numpy()

            if line2 is None:
                line2 = ax.scatter(y_plot, x_plot)
            else:
                line2.set_offsets(np.c_[y_plot, x_plot])
            fig.canvas.draw_idle()
            plt.pause(0.1)


    plt.waitforbuttonpress()