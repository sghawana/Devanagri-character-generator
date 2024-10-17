import torch

import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


def load_images_from_directory(root_dir):
    image_list = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(subdir, file)
                with Image.open(file_path) as img:
                    img_array = np.expand_dims(np.array(img), axis=0) #channel dimension
                    image_list.append(img_array)
    return image_list



class devnagari(Dataset):
    def __init__(self, data, device, dtype):
        super().__init__()
        self.data = data
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index]
        img_tensor = torch.tensor(img, dtype=self.dtype, device=self.device)
        return img_tensor
