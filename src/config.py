import os


class Config:
    WORKING_DIR = 'C:\\Users\\dhiae\\RANZ_kaggle'
    DATA_DIR = os.path.join(WORKING_DIR, 'data')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALIDATION_DIR = os.path.join(DATA_DIR, 'val')

    CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'checkpoint')

    TEXT_DIR = os.path.join(DATA_DIR, 'text')
    LABELS_FILE = os.path.join(DATA_DIR, 'data.csv')
    TRAIN_LABELS_FILE = os.path.join(DATA_DIR, 'train.csv')
    VAL_LABELS_FILE = os.path.join(DATA_DIR, 'val.csv')

    VALIDATION_RATIO = 0.15
    TRAIN_RATIO = 0.85
