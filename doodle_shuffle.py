from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import cv2
import ast

class DoodlesShuffleDataset(Dataset):
    """Doodles csv dataset."""

    def __init__(self, csv_file, root_dir, nrows, mode='train', size=256, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            nrows (int): Number of rows of file to read. Useful for reading pieces of large files.
            mode (string): Train or test mode.
            size (int): Size of output image.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        file = os.path.join(self.root_dir, csv_file)
        self.size = size
        self.mode = mode
        self.doodle = pd.read_csv(file, nrows=nrows)
        self.transform = transform

    @staticmethod
    def _draw(raw_strokes, size=256, lw=6, time_color=True):
        BASE_SIZE = 256
        img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
        for t, stroke in enumerate(raw_strokes):
            for i in range(len(stroke[0]) - 1):
                color = 255 - min(t, 10) * 13 if time_color else 255
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                             (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
        if size != BASE_SIZE:
            return cv2.resize(img, (size, size))
        else:
            return img

    @staticmethod
    def _get_label(en_dict, nfile):
        """ Return encoded label for class by name of csv_files """
        return en_dict[nfile.replace(' ', '_')[:-4]]

    def __len__(self):
        return len(self.doodle)

    def __getitem__(self, idx):
        raw_strokes = ast.literal_eval(self.doodle.drawing[idx])
        label = self.doodle.y[idx]
        sample = self._draw(raw_strokes, size=self.size, lw=2, time_color=True)
        if self.transform:
            sample = self.transform(sample)
        if self.mode == 'train':
            return (sample[None] / 255).astype('float32'), label
        else:
            return (sample[None] / 255).astype('float32')
