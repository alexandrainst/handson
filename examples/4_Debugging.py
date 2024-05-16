"""
This part is designed to teach you some things to do if training seems to fail.
    In general, it is good practice to use some of these tricks before training even begins, but sometimes stuff simply goes wrong anyway.

1) Typically you want to train a model many times with the full dataset.
    During debugging, we attempt to isolate the problem. In this case, that means overfitting to a single minibatch.

2) We use the structures and code generated during previous steps to make the experience uniform, but change the training loop for a simpler overfitting loop.
"""
from config.unet import Configuration
import torch
import matplotlib.pyplot as plt
from src.dataloader import load_data_wrapper
from src.losses import select_loss_fnc
from src.model import model_loader_wrapper


if __name__ == "__main__":
    conf = Configuration()
    dataloaders = load_data_wrapper(**conf)
    model = model_loader_wrapper(**conf)

    # we will start by setting up the optimization parameters
    loss_fnc = select_loss_fnc(**conf)
    optimizer = torch.optim.Adam(model.parameters(), lr=100)

    data, target = next(iter(dataloaders['train']))
    data = data.to(conf.device)
    target = target.to(conf.device)

    losses = []  # keep track of our losses

    for i in range(100):
        # print(i)  # sometimes it can be good to keep track of how far along we are

        # zero gradients
        optimizer.zero_grad()

        # calculate the model's output based on the data and sigmoid to transform to range 0-1
        output = model(data)
        output = output.sigmoid()

        # calculcate loss
        loss = loss_fnc(output, output)

        # track the loss
        losses.append(loss.item())

        # calculate gradients and update weights
        loss.backward

    plt.plot(losses)
    plt.show()





