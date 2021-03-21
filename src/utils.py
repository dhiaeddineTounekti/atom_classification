import os

import albumentations
import cv2
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from config import Config


def plot_tensor_images(tensor: Tensor) -> None:
    """
    Plots tensor images
    :param tensor:
    :return:
    """
    # Change from (C, H, W) -> (H, W, C)
    array = tensor.permute([1, 2, 0]).cpu().detach().numpy()
    plt.imshow(array)
    plt.show()


def compress_images():
    """
    Compress the images in there respective folders.
    :return:
    """
    conf = Config()
    data_dirs = [conf.TRAIN_DIR, conf.VALIDATION_DIR]

    for directory in data_dirs:
        for image_name in os.listdir(directory):
            image = cv2.imread(os.path.join(directory, image_name))
            image = albumentations.Resize(640, 640)(image=image)['image']
            cv2.imwrite(filename=os.path.join(directory, image_name.replace('.jpg', '_compressed.jpg')), img=image,
                        params=[int(cv2.IMWRITE_JPEG_QUALITY), 90])


def threshold(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Take the tensor and change the probabilities to binary tensor
    :param tensor:
    :param threshold:
    :return:
    """
    filter_1 = nn.Threshold(-threshold, value=-1)
    filter_0 = nn.Threshold(threshold, 0)
    return filter_0(-filter_1(tensor * (-1.0)))
