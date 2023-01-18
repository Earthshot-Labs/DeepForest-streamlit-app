import os
import glob
import random
import torch
from torchvision import transforms as transforms
from PIL import Image
import tifffile

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
        dataset_dir = os.path.join(opt.dataset_dir)
        format = opt.format
        format_input = 'tif'

        self.label_path_list = sorted(
            glob.glob(os.path.join(dataset_dir, '*.' + format_input)))

    def get_transform(self, normalize=True):
        transform_list = []

        transform_list.append(transforms.ToTensor())

        if normalize:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

        return transforms.Compose(transform_list)

    @staticmethod
    def __flip(x):
        return x.transpose(Image.FLIP_LEFT_RIGHT)

    def __getitem__(self, index):
        self.coin = None
        label_array = tifffile.imread(self.label_path_list[index])
        # label_array = Image.open(self.label_path_list[index])
        label_tensor = self.get_transform(normalize=True)(label_array)

        if self.opt.is_inference:
            return label_tensor

        target_array = Image.open(self.target_path_list[index])
        target_tensor = self.get_transform(normalize=True)(target_array)

        # input_tensor = self.encode_input(label_tensor)

        return label_tensor, target_tensor

    def __len__(self):
        return len(self.label_path_list)

