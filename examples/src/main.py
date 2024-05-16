import torch

from src.dataloader import load_data_wrapper
from src.losses import select_loss_fnc
from src.model import model_loader_wrapper
from src.optimizer import optimize_model
from src.viz import eval_unlabelled_images, plot_loss_and_accuracy


def main(configuration):
    """
    A full training and evaluation pipeline
    """

    dataloaders = load_data_wrapper(**configuration)
    model = model_loader_wrapper(**configuration)
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['lr'])
    loss_fnc = select_loss_fnc(**configuration)
    losses, accuracies = optimize_model(model, dataloaders, optimizer, loss_fnc, **configuration)
    plot_loss_and_accuracy(losses, accuracies)
    eval_unlabelled_images(model,configuration.path_input_test)




