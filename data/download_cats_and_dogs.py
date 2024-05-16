import glob
import os
import shutil
import zipfile
from os.path import exists
from pathlib import Path

from tqdm import tqdm
import requests

"""
A simple script that downloads a dataset from zenodo and extracts it.
"""

def download_dogs_vs_cats_data(filename, overwrite=False):

    already_downloaded = exists(filename)

    if already_downloaded and not overwrite:
        return
    else:
        response = requests.get("https://zenodo.org/records/10997322/files/dogs_vs_cats_subset.zip?download=1", stream=True)

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(filename, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")
        return


if __name__ == "__main__":
    filename = "dogs_vs_cats.zip"
    directory_to_extract_in = '.'

    download_dogs_vs_cats_data(filename)

    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_in)
    # path_root = './cats_dogs_light'
    # path_val = os.path.join(path_root,'val')
    # path_train = os.path.join(path_root,'train')
    # files_train_cat = glob.glob(os.path.join(path_train,"cat*.jpg"))
    # files_train_dog = glob.glob(os.path.join(path_train,"dog*.jpg"))
    # os.makedirs(path_val, exist_ok=True)
    # files_val_cat = glob.glob(os.path.join(path_val,"cat*.jpg"))
    # files_val_dog = glob.glob(os.path.join(path_val,"dog*.jpg"))
    # if files_train_cat == 500 and files_train_dog == 500 and files_val_cat == 0 and files_val_dog == 0:
    #     print("moving files from training dataset, to validation set...")
    #     files_cat_to_move = files_train_cat[:50]
    #     files_dog_to_move = files_train_dog[:50]
    #     for file in files_cat_to_move:
    #         file_dest = os.path.join(path_val,Path(file).stem+".jpg")
    #         shutil.move(file, file_dest)
    #     for file in files_dog_to_move:
    #         file_dest = os.path.join(path_val,Path(file).stem+".jpg")
    #         shutil.move(file, file_dest)
    #     print("Done, moving files to validation dataset.")
    # elif files_train_cat == 450 and files_train_dog == 450 and files_val_cat == 50 and files_val_dog == 50:
    #     print("validation set already created.")
    # else:
    #     raise FileExistsError("The amount of files in train and validation folder deviates from the expected numbers.")
    #
    #



