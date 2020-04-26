import os, glob
import pandas as pd
import cv2
import json
import numpy as np
import shutil

DATA_DIR = os.path.join(os.getcwd(), 'input/train_simplified/')
DIR_NAME = "doodle_images"
print(DATA_DIR)

filenames = glob.glob(os.path.join(DATA_DIR, '*.csv'))
filenames = sorted(filenames)

os.mkdir(DIR_NAME)


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

for filename in filenames:
    print("Now deal with " + filename)
    category = filename[:-4].split('/')[-1].replace(' ', '_')
    saved_dir = DIR_NAME + '/' + category
    os.mkdir(saved_dir)
    file = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(file, usecols=["drawing"])
    df["drawing"] = df["drawing"].apply(json.loads)
    for i, raw_strokes in enumerate(df.drawing.values):
        nd = _draw(raw_strokes, size=64, lw=2,
                   time_color=False)
        cv2.imwrite(saved_dir + "/" + str(i) + ".jpg", nd)
    shutil.make_archive(saved_dir, 'zip', saved_dir)
    shutil.rmtree(saved_dir)
