import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import config as c
from data_loader import mri_DataLoader
from model import ResNet

if __name__ == "__main__":
    config = c.Config
    if not os.path.isdir(config.CHECKPOINT_DIR):
        os.mkdir(config.CHECKPOINT_DIR)

    model = ResNet()
    dataset = mri_DataLoader()
    check_point_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        monitor='val_loss',
        filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )
    trainer = Trainer(callbacks=[early_stop_callback, check_point_callback], gpus=1)
    trainer.fit(model=model, datamodule=dataset)
