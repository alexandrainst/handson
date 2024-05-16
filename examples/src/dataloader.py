import glob
import os
from random import shuffle

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

def load_data_wrapper(path_input_train,path_input_val,path_input_test,n_train,n_val,n_test,batch_size_train,batch_size_val,batch_size_test, device,**kwargs):
    """
    A wrapper that allows us to handle different datasets.
    """

    dataloaders = load_cat_dog_data(path_input_train,path_input_val,path_input_test,n_train,n_val,n_test,batch_size_train,batch_size_val,batch_size_test, device)

    return dataloaders


def load_cat_dog_data(path_input_train,path_input_val,path_input_test,n_train,n_val,n_test,batch_size_train,batch_size_val,batch_size_test, device):
    dataloaders = {}

    dataset_train = ImageDataset(path_input_train,n_train,device=device)
    dataloaders['train'] = DataLoader(dataset_train,shuffle=True, batch_size=batch_size_train,drop_last=True)

    dataset_val = ImageDataset(path_input_val,n_val,device=device)
    dataloaders['val'] = DataLoader(dataset_val,shuffle=False, batch_size=batch_size_val)

    dataset_test = ImageDataset(path_input_test,n_test,device=device,training_data=False)
    dataloaders['test'] = DataLoader(dataset_test,shuffle=False, batch_size=batch_size_test)
    return dataloaders

def eval_transform():
    """
    A transform that crops an image to size of 256x256 converts it to a torch tensor
    and normalizes the data according to the imagenet dataset.
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(256),  # crops the image in the center
        torchvision.transforms.ToTensor(), # This permutes the dimensions of the image such that they are ordered the way neural networks usually works with them and then converts their datatype into floats or doubles.
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

class ImageDataset(data.Dataset):
    """
    Our custom dataset used for cats and dogs images.
    """

    def __init__(self, path,n_samples=-1,extension='jpg',transform=eval_transform(),training_data=True,device='cpu'):
        self.path = path
        self.n_samples = n_samples
        self.transform = transform
        self.training_data = training_data
        self.device = device
        search_param = os.path.join(path,f"*.{extension}")
        files = glob.glob(search_param)
        assert len(files)>0, f"Searching for images files in {search_param} gave zero hits. Make sure you already downloaded the dataset using /data/download_cats_and_dogs.py"
        shuffle(files) # Remember to shuffle the pictures before you extract a subset of them!
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
                label = 0
            elif "dog" in name:
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


