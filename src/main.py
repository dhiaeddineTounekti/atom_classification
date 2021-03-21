import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler

import config as c
from data_loader import custom_DataLoader
from model import ResNet

if __name__ == "__main__":
    config = c.Config
    if not os.path.isdir(config.CHECKPOINT_DIR):
        os.mkdir(config.CHECKPOINT_DIR)

    model = ResNet()
    dataset = custom_DataLoader()
    logger = TensorBoardLogger("tb_logs", name="Resnet50", log_graph=True)
    check_point_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        monitor='val_loss',
        filename='ResNet50-{epoch:02d}-{val_loss:.2f}',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min'
    )
    profiler = AdvancedProfiler(output_filename=os.path.join(config.CHECKPOINT_DIR, 'batch_accumulation'))
    trainer = Trainer(
                      callbacks=[early_stop_callback, check_point_callback], gpus='0', logger=logger, profiler=profiler)
    trainer.fit(model=model, datamodule=dataset)
    model = ResNet.load_from_checkpoint(checkpoint_path='C:\\Users\\dhiae\\RANZ_kaggle\\checkpoint\\ResNet200d-epoch=20-val_loss=0.20.ckpt')
    trainer.test(model=model, datamodule=dataset)

