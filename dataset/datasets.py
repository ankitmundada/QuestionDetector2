"""
Various classes to load data of different sources/formats etc
"""


from skimage import io
import numpy as np
from os.path import join
import pandas as pd
from torch.utils.data import Dataset


class FER2013Dataset(Dataset):
    """
    FER2013 Dataset
    """

    def __init__(self, data_folder, transform=None):
        """
        """
        self.transform = transform
        self.data_folder = data_folder
        manifest_path =  join(data_folder, "manifest.csv")
        self.manifest = pd.read_csv(manifest_path)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        img_name = self.manifest.iloc[idx, 0]
        img_path = join(self.data_folder, img_name)
        img = io.imread(img_path)

        img = img.reshape((48, 48, 1))
        label = self.manifest.iloc[idx, 1]

        item = {"image": img, "label": label}

        if self.transform:
            item = self.transform(item)

        return item


