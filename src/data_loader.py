import os
import shutil
from typing import Optional

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
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(60),
            T.RandomAffine(degrees=(0, 90), translate=(10, 10), shear=(10, 20)),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return DataLoader(
            dataset=mri_dataset(df=self.train, transform=transform, load_in_memory=True,
                                directory=self.config.TRAIN_DIR),
            shuffle=True, batch_size=64)

    def val_dataloader(self):
        transform = ToTensorV2()
        return DataLoader(dataset=mri_dataset(df=self.val, load_in_memory=True, directory=self.config.VALIDATION_DIR,
                                              transform=transform),
                          batch_size=64)
