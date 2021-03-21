import os

import albumentations as A
import cv2
import numpy
import torch
from torch.utils.data import Dataset


class mri_dataset(Dataset):
    def __init__(self, df, directory, transform=None, load_in_memory=False, image_size=None, aug_on_gpu=False):
        if image_size is None:
            image_size = [224, 224]
        self.transform = transform
        self.image_names = df.iloc[:, 1].values
        self.labels = df.iloc[:, 2:13].values
        self.data_dir = directory
        self.pre_transformations = A.Compose([
            A.CenterCrop(100, 100)
        ])
        self.load_in_memory = load_in_memory
        self.aug_on_gpu = aug_on_gpu
        if load_in_memory:
            self.images = self.read_all_images_in_memory()

    def read_all_images_in_memory(self):
        compression_param = [cv2.IMWRITE_JPEG_QUALITY, 90]
        result = []
        count = 0
        total_memory_consumption = 0
        for image_id in self.image_names:
            image = cv2.imread(os.path.join(self.data_dir, str(image_id) + '.jpg'))
            image = self.pre_transformations(image=image)

            # Encode image
            image = cv2.imencode('.jpg', image['image'], compression_param)[1]
            result.append(image)
            total_memory_consumption += image.size

            count += 1
            if count % 200 == 0:
                print(f'Loading images: {count}/{len(self.labels)}')

        print(f'Total memory consumption: {total_memory_consumption / (1024 * 1024 * 1024)} GB')
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_id = self.image_names[idx]

        if self.load_in_memory:
            image = cv2.imdecode(self.images[idx], 1)
        else:
            image = cv2.imread(os.path.join(self.data_dir, str(image_id) + '.jpg'), cv2.IMREAD_UNCHANGED)

        target = self.labels[idx].astype(numpy.float32)

        if self.aug_on_gpu:
            image = torch.FloatTensor(image)
            target = torch.FloatTensor(target)

        if self.transform is not None:
            image = self.transform(image)

        return image, target, image_id
