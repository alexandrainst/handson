import glob
import os

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F

from src.dataloader import eval_transform

def plot_loss_and_accuracy(losses, accuracies):
    fig, axs = plt.subplots(1, 2, figsize=(8*2, 5))
    axs[0].semilogy(losses['train'], label='train')
    axs[0].semilogy(losses['val'],label='val')
    axs[0].set_title(f"loss as a function of epochs")
    axs[0].legend()
    axs[0].set(xlabel='Epoch',ylabel='Loss')
    axs[1].plot(accuracies['train'], label='train')
    axs[1].plot(accuracies['val'], label='val')
    axs[1].set_title(f"Accuracy as a function of epochs"),
    axs[1].legend()
    axs[1].set(xlabel='Epoch',ylabel='Accuracy')
    plt.show()


def eval_unlabelled_images(model,path, device='cpu', extension='jpg'):
    """
    All the imagefiles with the given extension in the path are run through the model one at a time and then the original image,
    as well as the image that the model is seeing are shown along with the model prediction.
    """
    search_param = os.path.join(path, f"*.{extension}")
    files = glob.glob(search_param)
    model.train(False)
    model.to(device=device)
    torch.set_grad_enabled(False)
    transform = eval_transform()
    label_names = ['cat', 'dog']
    fig, axs = plt.subplots(1,2, figsize=(10,10))
    for file in files:
        image = Image.open(file)
        image_for_model = transform(image)[None].to(device=device)

        prediction_prob = model(image_for_model)
        prediction = torch.argmax(prediction_prob, dim=1)
        axs[0].imshow(image)
        axs[0].set_title(f"Original image")
        image_transformed = (image_for_model[0].cpu().permute(1, 2, 0) * 255).to(dtype=torch.uint8)
        axs[1].imshow(image_transformed)
        axs[1].set_title(f"Predicted Label={label_names[prediction]}")
        plt.pause(5)
    return