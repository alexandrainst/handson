"""
Our current approach load each image when needed from disk to memory.
By using pytorch built-in dataloader we get access to their powerful prefetcher, which can assign a set of workers to prefetch data for the model such that the next examples are always available in memory when needed.
It also allows memory pinning and various other advanced options for more information look at https://pytorch.org/docs/stable/data.html
"""
import glob
import os

import torch
import torchvision
import torch.utils.data as data
from PIL import Image

from src.dataloader import eval_transform


train_transform = torchvision.transforms.Compose([  # sequential transformation, lots of randomness here
    torchvision.transforms.RandomRotation(180),
    torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    torchvision.transforms.Resize(336),
    torchvision.transforms.RandomCrop(256, pad_if_needed=True),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # this is the ImageNet normalization parameters, so our model is pretrained on this
])



class ImageDataset(data.Dataset):
    """
    Our custom dataset used for cats and dogs images. Now with additional safety check of filenames.
    """

    def __init__(self, path,n_samples=-1,extension='jpg',transform=eval_transform(),training_data=True,device='cpu'):
        self.path = path
        self.n_samples = n_samples
        self.transform = transform
        self.training_data = training_data
        self.device = device
        search_param = os.path.join(path,f"*.{extension}")
        files = glob.glob(search_param)
        self.files = files[:n_samples]
        self.label_names = ['cat', 'dog']
        return

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + ')'

    def __getitem__(self, index):
        file = self.files[index]
        name = file.split('/')[-1]
        if self.training_data:
            if "cat" in name:
                assert "dog" not in name, f"{file} has both cat and dog in its name."
                label = 0
            elif "dog" in name:
                assert "cat" not in name, f"{file} has both cat and dog in its name."
                label = 1
            else:
                raise ValueError(f"{file} does not contain dog or cat label.")
        else:
            label = -1

        # load image
        image = Image.open(file)

        if self.transform:
            image = self.transform(image)

        image = image.to(device=self.device)
        label = torch.tensor(label,device=self.device,dtype=torch.int64)
        return image, label


