import os
import shutil
from typing import Optional

import medicaltorch.transforms as mt
import pandas as pd
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import config as c
from dataset import mri_dataset


class mri_DataLoader(LightningDataModule):

    def __init__(self):
        super(mri_DataLoader, self).__init__()
        self.config = c.Config()

    def setup(self, stage: Optional[str] = None):
        self.val = pd.read_csv(self.config.VAL_LABELS_FILE)
        self.train = pd.read_csv(self.config.TRAIN_LABELS_FILE)

    def prepare_data(self):
        if not os.path.isfile(self.config.VAL_LABELS_FILE) or not os.path.isfile(self.config.TRAIN_LABELS_FILE):
            dataset = pd.read_csv(self.config.LABELS_FILE)

            self.train = dataset.sample(frac=self.config.TRAIN_RATIO)
            self.val = dataset.drop(self.train.index)

            # Move data to val folder
            print('Moving data...')
            if not os.path.isdir(self.config.VALIDATION_DIR):
                os.mkdir(self.config.VALIDATION_DIR)

            for image_id in self.val.iloc[:, 0]:
                image_path = os.path.join(self.config.TRAIN_DIR, image_id + '.jpg')
                shutil.move(image_path, self.config.VALIDATION_DIR)

            self.val.to_csv(self.config.VAL_LABELS_FILE)
            self.train.to_csv(self.config.TRAIN_LABELS_FILE)

    def train_dataloader(self):
        transform = T.Compose([
            T.ToTensor(),
            # T.Resize(size=(640, 640)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop(size=640, scale=(0.8, 1.0)),
            T.RandomAffine(degrees=(0, 90)),
            mt.NormalizeInstance()
        ])
        return DataLoader(
            dataset=mri_dataset(df=self.train, transform=transform, directory=self.config.TRAIN_DIR)
            , num_workers=1, batch_size=32, persistent_workers=True, prefetch_factor=1, pin_memory=True)

    def val_dataloader(self):
        transform = T.Compose([
            T.ToTensor(),
            # T.Resize(size=(640, 640)),
            mt.NormalizeInstance()
        ])
        return DataLoader(dataset=mri_dataset(df=self.val, directory=self.config.VALIDATION_DIR,
                                              transform=transform), num_workers=1, batch_size=32,
                          persistent_workers=True)
