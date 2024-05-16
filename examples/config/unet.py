from dataclasses import dataclass
from typing import Any

import torch
import torchvision
from collections.abc import Mapping


@dataclass(frozen=True)
class Configuration(Mapping):
    #General
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # device: str = "cpu"

    #Data
    path_input_train: str = "./../data/dogs_vs_cats_subset/train"
    path_input_val: str = "./../data/dogs_vs_cats_subset/val"
    path_input_test: str = "./../data/dogs_vs_cats_subset/test"
    n_train: int = 100
    n_val: int = 100
    n_test: int = 100
    batch_size_train: int = 5
    batch_size_val: int = 10
    batch_size_test: int = 5


    #Model
    model_name: str = 'unet'
    model_weights: Any = None

    #Optimization
    epochs: int = 5
    lr: float = 0.01
    loss_type: str = 'cross_entropy'

    def __getitem__(self, x):
        return self.__dict__[x]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)
