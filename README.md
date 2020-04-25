# quickdraw
A simple quickdraw training code

Using PyTorch with pretrained MobileNet V2.

Please see the [reference implmentation](https://www.kaggle.com/leighplt/pytorch-starter-kit/data).

## Setup

You can download the dataset through kaggle:

```
kaggle competitions download -c quickdraw-doodle-recognition
```

The data size is very big, around 76GB. In our use case, we just need `train_simplified` folder (everything unzipped is arond 400 GB).

```
unzip quickdraw-doodle-recognition.zip train_simplified/* -d /input
```
This will download the csv files in the input folder.


You can find more information about the data here:

[QuickDraw challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition/data)

## Start training

Simply run with
```
train.py
```
 
Note, we expand the grayscale image into 3 dimensions to RGB. This will help to fit the MobileNet internal architecture. The optimization can be taken to boost the training performance. Training speed on CPU is very low, please try to train on a GPU.
