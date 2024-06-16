import os
import torch
from torchvision import transforms
from skimage import io
from torch.utils.data import Dataset
import random


class ParcelDataset(Dataset):
    def __init__(self, txt_path, image_root, label_root, data_augmentation=True):
        super().__init__()

        self.data_augmentation = data_augmentation

        assert os.path.exists(txt_path) is True, 'txt_path not exist: %s' % txt_path
        files = []

        with open(txt_path, 'r') as f:
            for i in f:
                i = i.strip()
                save_path_image = os.path.join(image_root, i)
                save_path_label = os.path.join(label_root, i)
                files.append((save_path_image, save_path_label))
        f.close()
        self.files = files
        # print(len(files), files[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        path_image, path_label = self.files[item]
        image = io.imread(path_image) / 255.  # (h, w, c)
        label = io.imread(path_label) / 255.

        assert image.shape[1] == label.shape[1]

        # to tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        label = torch.from_numpy(label).float().unsqueeze(dim=0)

        if self.data_augmentation:
            image, label = self._data_augmentation(image, label)

        return image, label

    @staticmethod
    def _data_augmentation(image, label):
        c1, c2 = image.shape[0], label.shape[0]
        data = torch.cat((image, label), 0)

        trans1 = transforms.RandomChoice([transforms.RandomRotation(degrees=[0, 0]),
                                          transforms.RandomHorizontalFlip(p=1),
                                          transforms.RandomVerticalFlip(p=1),
                                          transforms.RandomRotation(degrees=[90, 90]),
                                          transforms.RandomRotation(degrees=[180, 180])])
        data = trans1(data)

        image, label = data.split([c1, c2], 0)

        return image, label
