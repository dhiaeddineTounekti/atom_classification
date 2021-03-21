import os
from typing import Optional, Union, List

import medicaltorch.transforms as mt
import pandas as pd
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import config as c
from dataset import mri_dataset


class custom_DataLoader(LightningDataModule):

    def __init__(self):
        super(custom_DataLoader, self).__init__()
        self.config = c.Config()

    def setup(self, stage: Optional[str] = None):
        self.val = pd.read_csv(self.config.VAL_LABELS_FILE)
        self.train = pd.read_csv(self.config.TRAIN_LABELS_FILE)

    def prepare_data(self):
        if not os.path.isfile(self.config.VAL_LABELS_FILE) or not os.path.isfile(self.config.TRAIN_LABELS_FILE):
            dataset = pd.DataFrame([[os.path.join(self.config.DATA_DIR, path), 0 if path.split('_')[5] == 'ER' else 1,
                                     float(path.split('_')[6]) if path.split('_')[5] == 'ER' else float(
                                         path.split('_')[7])] for path in os.listdir(self.config.DATA_DIR)])

            self.train = dataset.sample(frac=self.config.TRAIN_RATIO)
            self.val = dataset.drop(self.train.index)

            self.val.to_csv(self.config.VAL_LABELS_FILE)
            self.train.to_csv(self.config.TRAIN_LABELS_FILE)

    def train_dataloader(self):
        transform = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=(0, 90)),
            mt.NormalizeInstance()
        ])
        return DataLoader(
            dataset=mri_dataset(df=self.train, transform=transform, directory=self.config.TRAIN_DIR)
            , num_workers=4, batch_size=64, persistent_workers=True)

    def val_dataloader(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=(224, 224)),
            mt.NormalizeInstance()
        ])
        return DataLoader(dataset=mri_dataset(df=self.val, directory=self.config.VALIDATION_DIR,
                                              transform=transform), num_workers=2, batch_size=64,
                          persistent_workers=True)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=(224, 224)),
            mt.NormalizeInstance()
        ])
        return DataLoader(dataset=mri_dataset(df=self.val, directory=self.config.VALIDATION_DIR,
                                              transform=transform), num_workers=5, batch_size=64,
                          persistent_workers=True)
