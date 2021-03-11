import os
import pickle
import shutil

from numpy import random

import config

conf = config.Config()


def generate_captions(labels_path: str):
    """
    Generates 5 classes_in_captions from labels for each mri.
    :param labels_path: the path to the labels path
    :return: None
    """
    labels_file = open(labels_path, 'r')
    # Crop first line that contains the headers
    all_labels = labels_file.readlines()[1:]
    for row in all_labels:
        columns = row.split(',')
        mri_id = columns[0]
        mri_labels = columns[1:-1]

        class_names = ['an abnormal ETT', 'a borderline ETT', 'a normal ETT', 'an abnormal NGT', 'a borderline NGT',
                       'an incompletely imaged NGT', 'a normal NGT', 'an abnormal CVC', 'a borderline CVC',
                       'a normal CVC', 'a Swan Ganz Catheter']
        caption_beginning = ['This patient presents', 'This MRI shows', 'In this medical record we have']

        # If label i of mri labels is equal to 1 then include class_name i inside the caption.
        classes_in_captions = [class_names[index] for index in range(len(mri_labels)) if int(mri_labels[index])]
        captions = []
        for iteration in range(5):
            if len(classes_in_captions) > 1:
                captions.append(
                    f'{caption_beginning[iteration % 3]} {",".join(classes_in_captions[:-2])} and {classes_in_captions[-1]}.\n')
            elif len(classes_in_captions) == 1:
                captions.append(f'{caption_beginning[iteration % 3]} {classes_in_captions[0]}.\n')
            else:
                captions.append(f'{caption_beginning[iteration % 3]} nothing.\n')

        caption_file = open(os.path.join(conf.TEXT_DIR, mri_id + '.txt'), 'w')
        caption_file.writelines(captions)
        caption_file.close()


def split_data(data_path: str, val_data_path: str):
    """
    split the data into validation and train data using the config split ratio
    :param data_path: path to the data folder
    :param val_data_path: path to the test folder
    :return:
    """
    assert os.path.isdir(data_path), f'Folder {data_path} does not exist'
    # Create validation set directory if it does not exist
    if not os.path.isdir(val_data_path):
        os.mkdir(val_data_path)

    all_data = os.listdir(data_path)
    random.shuffle(all_data)
    val_data = all_data[: int(len(all_data) * conf.VALIDATION_RATIO)]

    # Move the validation images to the corresponding folder.
    print('splitting data ...')
    for file_name in val_data:
        shutil.move(os.path.join(data_path, file_name), val_data_path)


def generate_filenames(train_dir: str, validation_dir: str):
    """
    Generate two pickle files containing the file names of validation and training samples.
    :param train_dir: train simples directory
    :param validation_dir: validation simples directory
    :return:
    """
    val_samples = [".".join(file_name.split('.')[:-1]) for file_name in os.listdir(validation_dir) if
                   file_name.split('.')[-1] != 'pickle']
    train_samples = [".".join(file_name.split('.')[:-1]) for file_name in os.listdir(train_dir) if
                     file_name.split('.')[-1] != 'pickle']

    val_file_path = open(os.path.join(validation_dir, 'filenames.pickle'), 'wb')
    train_file_path = open(os.path.join(train_dir, 'filenames.pickle'), 'wb')

    # Dump the files where they should be.
    print('Generating filenames.pickle files ...')
    pickle.dump(val_samples, val_file_path)
    pickle.dump(train_samples, train_file_path)


if __name__ == "__main__":
    # generate_captions(conf.LABELS_FILE)

    # split_data(conf.TRAIN_DIR, conf.VALIDATION_DIR)
    generate_filenames(conf.TRAIN_DIR, conf.VALIDATION_DIR)
