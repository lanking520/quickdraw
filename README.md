# quickdraw
A simple quickdraw training code

Using PyTorch with MobileNet V2. I customized the MobileNet V2 by using 1 channel.

Please see the [reference implmentation](https://www.kaggle.com/leighplt/pytorch-starter-kit/data).

## Setup

You can download the dataset through kaggle:

```
kaggle competitions download -c quickdraw-doodle-recognition
```

The data size is very big, around 76GB. In our use case, we just need `train_simplified` folder (everything unzipped is arond 400 GB).

```
unzip quickdraw-doodle-recognition.zip train_simplified/* -d input/
```
This will download the csv files in the input folder.

After this step, I converted csv to images using `csv2img.py`. This step takes very long and the data size is around 200GB.

You can find more information about the data here:

[QuickDraw challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition/data)

### Shuffled dataset

It may not be necessary to use the big dataset explained before, you can download this instead:

[shuffled_csv](https://www.kaggle.com/gaborfodor/shuffle-csvs)

The size is 1.7 GB but contains 8.16M data.

## Start training

Simply run with
```
train_image.py
```

for Shuffled csv:

```
train_shuffle_batch.py
```

Training speed on CPU is very low, please try to train on a GPU (change `DEVICE` to `cuda`).

## Result

On small dataset (shuffled CSV):

Top 1 Accuracy is 91% (15 Epoch) and 79% for top 1.
