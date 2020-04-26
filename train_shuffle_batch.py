import torch
import os
import glob
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader

from doodle_random import DoodlesRandomDataset
from mobilenet_grayscale import get_MobileNet_grayscale

DATA_DIR = os.path.join(os.getcwd(), 'input/shuffle-csvs/')
print(DATA_DIR)
BASE_SIZE = 64 # Use new base size
NCSVS = 100 # num csv files
NCATS = 340 # num classes
STEPS = 1000
BATCH_SIZE = 256
EPOCHS = 16
DEVICE = "cpu"
np.random.seed(seed=1987)
torch.manual_seed(1987)

filenames = glob.glob(os.path.join(DATA_DIR, '*.csv.gz'))
filenames = sorted(filenames)

doodles = ConcatDataset([DoodlesRandomDataset(fn.split('/')[-1], DATA_DIR, chunksize=BATCH_SIZE, size=BASE_SIZE) for fn in filenames])

print('Train set:', len(doodles))

loader = DataLoader(doodles, batch_size = 1, num_workers = 0)

def accuracy(output, target, topk=(3,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def mapk(output, target, k=3):
    """
    Computes the mean average precision at k.

    Parameters
    ----------
    output (torch.Tensor): A Tensor of predicted elements.
                           Shape: (N,C)  where C = number of classes, N = batch size
    target (torch.int): A Tensor of elements that are to be predicted.
                        Shape: (N) where each value is  0≤targets[i]≤C−1
    k (int, optional): The maximum number of predicted elements

    Returns
    -------
    score (torch.float):  The mean average precision at k over the output
    """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i in range(k):
            correct[i] = correct[i] * (k - i)

        score = correct[:k].view(-1).float().sum(0, keepdim=True)
        score.mul_(1.0 / (k * batch_size))
        return score


model = get_MobileNet_grayscale(NCATS, pretrained=False)
if os.path.exists('checkpoint_mobilenet.pth'):
    print("Found checkpoint file, continuing training...")
    model.load_state_dict(torch.load('checkpoint_mobilenet.pth'))
if DEVICE is "cuda":
    model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = torch.nn.CrossEntropyLoss()

lsize = len(loader)
itr = 1
model.train()
tloss, score = 0, 0
for epoch in range(EPOCHS):
    print("epoch " + str(epoch) + " start")
    for x, y in loader:
        if len(x.shape) != 3:
            # Skip single or empty data
            continue
        x = x.squeeze().unsqueeze(1)
        y = y.squeeze()
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        tloss += loss.item()
        score += mapk(output, y)[0].item()
        if itr % STEPS == 0:
            print('Epoch {} Iteration {} -> Train Loss: {:.4f}, MAP@3: {:.3f}'.format(epoch, itr, tloss / STEPS,
                                                                                      score / STEPS))
            tloss, score = 0, 0
        itr += 1
    filename_pth = 'checkpoint' + str(epoch) + '_mobilenet.pth'
    torch.save(model.state_dict(), filename_pth)


