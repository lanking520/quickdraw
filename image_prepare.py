import glob
import os
import shutil

DATA_DIR = "doodle_images/"
INPUT_DIR = "input/images"

filenames = glob.glob(os.path.join(DATA_DIR, '*.zip'))
filenames = sorted(filenames)

for filename in filenames:
    category = filename[:-4].split('/')[-1]
    saved_dir = os.path.join(INPUT_DIR + "/" + category)
    os.makedirs(saved_dir)
    shutil.unpack_archive(filename, saved_dir, "zip")

shutil.rmtree(DATA_DIR)
