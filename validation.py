import os

import torch
from torch.utils.data import DataLoader
from doodle_random import DoodlesRandomDataset
from mobilenet_grayscale import get_MobileNet_grayscale

DATA_DIR = 'input/shuffle-csvs/'
BATCH_SIZE = 256
BASE_SIZE = 64
NCATS = 340
DEVICE = "cpu"

valid_set = DoodlesRandomDataset("train_k80.csv.gz", DATA_DIR, chunksize=BATCH_SIZE, size=BASE_SIZE)
valid_loader = DataLoader(valid_set, batch_size=1, num_workers=0)

model = get_MobileNet_grayscale(NCATS, pretrained=False)
model.load_state_dict(torch.load('models/checkpoint15_mobilenet.pth',  map_location=torch.device('cpu')))
criterion = torch.nn.CrossEntropyLoss()

if DEVICE is "cuda":
    model.to(DEVICE)


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


def validation(model, valid_loader, device, lossf, scoref):
    model.eval()
    loss, score = 0, 0
    vlen = len(valid_loader)
    print("Validation data length " + str(vlen))
    for x, y in valid_loader:
        x = x.squeeze()
        if len(x.shape) != 3:
            # Skip single or empty data
            continue
        x = x.unsqueeze(1)
        y = y.squeeze()
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss += lossf(output, y).item()
        score += scoref(output, y)[0].item()
    model.train()
    return loss / vlen, score / vlen


vloss, vscore = validation(model, valid_loader, DEVICE, criterion, mapk)

print("VLoss: {}, Map@3 Score: {}".format(vloss, vscore))
