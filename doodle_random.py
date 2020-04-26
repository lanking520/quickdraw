from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import cv2
import json

class DoodlesRandomDataset(Dataset):
    """Doodles csv dataset."""

    def __init__(self, csv_file, root_dir, chunksize, mode='train', size=256, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            chunksize (int): chunk size per dataframe
            mode (string): Train or test mode.
            size (int): Size of output image.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        file = os.path.join(self.root_dir, csv_file)
        self.size = size
        self.mode = mode
        self.doodles = []
        for doodle in pd.read_csv(file, chunksize=chunksize, usecols=["drawing", "y"]):
            doodle["drawing"] = doodle["drawing"].apply(json.loads)
            self.doodles.append(doodle)

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

    def __len__(self):

        return len(self.doodles)

    def __getitem__(self, idx):
        doodle = self.doodles[idx]
        # form the batch
        x = np.zeros((len(doodle), 1, self.size, self.size))
        for i, raw_strokes in enumerate(doodle.drawing.values):
            x[i, 0] = self._draw(raw_strokes, size=self.size, lw=2,
                                       time_color=False)
        label = doodle.y.to_numpy()

        if self.transform:
            x = self.transform(x)
        if self.mode == 'train':
            return (x[None] / 255).astype('float32'), label
        else:
            return (x[None] / 255).astype('float32')
