import torch.nn
import torchvision

from src.model_unet import UNet


def model_loader_wrapper(model_name, model_weights=None, device='cpu', **kwargs):
    """
    Selects the model we want to use.

    Note that we have both premade/pretrained models as well as our own custom models in here.
    """

    if model_name == "resnet50":
        model = torchvision.models.resnet50(weights=model_weights)
    elif model_name == "efficient_net_b2":
        model = torchvision.models.efficientnet_b2(weights=model_weights)
        model.classifier[1] = torch.nn.Linear(1408,2,bias=True)
    elif model_name == 'unet':
        model = UNet()
    else:
        raise NotImplementedError(f"{model_name} has not been implemented.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{model_name} loaded, containing {total_params/1000000:2.2f}M trainable parameters.')
    model.to(device=device)

    return model

