import os
import os.path
from torch.utils.data import Dataset
import torch
import torch.utils
import torch.utils.data
from snntorch import spikegen
import numpy as np
import struct
from array import array

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

class MNISTCustomDataset(Dataset):
    def __init__(self, folder, train = True, num_steps = 1, transform=None):
        self.transform = transform
        self.train = train
        self.raw_folder = folder
        
        if num_steps < 1:
            raise Exception('num_steps must be more than 0')
        self.num_steps = num_steps

        self.images, self.labels = self._load_data()
    
    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = self.read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = self.read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets
    
    def read_image_file(self, images_filepath):        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())     

        images = []

        for i in range(size):
            images.append([0] * rows * cols)
        
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img    

        images = np.array(images)
        images = torch.Tensor(images)

        return images
    
    def read_label_file(self, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())
        
        labels = torch.Tensor(labels).type(torch.int64)

        return labels 

    def __getitem__(self, idx):
        label = self.labels[idx]
        x = self.images[idx]

        image = (x-torch.min(x))/(torch.max(x)-torch.min(x))
        image = image.view(-1)

        image = spikegen.rate(image, num_steps=self.num_steps)

        return image, label
    
    def __len__(self):
        return len(self.labels)