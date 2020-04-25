import torch
import os
import glob
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader

from doodle import DoodlesDataset
from mobilenet_grayscale import get_MobileNet_grayscale

DATA_DIR = os.path.join(os.getcwd(), 'input/train_simplified/')
print(DATA_DIR)
BASE_SIZE = 224  # Use pre-trained size
NCATS = 340  # num classes
DEVICE = "cpu"
np.random.seed(seed=1987)
torch.manual_seed(1987)

en_dict = {}
filenames = glob.glob(os.path.join(DATA_DIR, '*.csv'))
filenames = sorted(filenames)


def encode_files(filenames):
    """ Encode all label by name of csv_files """
    counter = 0
    for fn in filenames:
        en_dict[fn[:-4].split('/')[-1].replace(' ', '_')] = counter
        counter += 1


# collect file names and encode label
encode_files(filenames)
dec_dict = {v: k for k, v in en_dict.items()}


def decode_labels(label):
    return dec_dict[label]


# collect all single csvset in one
select_nrows = 10000
doodles = ConcatDataset([DoodlesDataset(fn.split('/')[-1], DATA_DIR, en_dict,
                                        nrows=select_nrows, size=BASE_SIZE) for fn in filenames])

print('Train set:', len(doodles))
# Use the torch dataloader to iterate through the dataset
loader = DataLoader(doodles, batch_size=128, shuffle=True, num_workers=2)


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


def validation(model, valid_loader, device, lossf, scoref):
    model.eval()
    loss, score = 0, 0
    vlen = len(valid_loader)
    for x, y in valid_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss += lossf(output, y).item()
        score += scoref(output, y)[0].item()
    model.train()
    return loss / vlen, score / vlen


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

model = get_MobileNet_grayscale(NCATS)
if os.path.exists('checkpoint_mobilenet.pth'):
    print("Found checkpoint file, continuing training...")
    model.load_state_dict(torch.load('checkpoint_mobilenet.pth'))
if DEVICE is "cuda":
    model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
criterion = torch.nn.CrossEntropyLoss()

# PyTorch scheduler:
# https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 12000, 18000], gamma=0.5)

epochs = 5
lsize = len(loader)
itr = 1
steps = 1000  # print every N iteration
model.train()
tloss, score = 0, 0
for epoch in range(epochs):
    print("epoch " + str(epoch) + " start")
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        tloss += loss.item()
        score += mapk(output, y)[0].item()
        scheduler.step()
        if itr % steps == 0:
            print('Epoch {} Iteration {} -> Train Loss: {:.4f}, MAP@3: {:.3f}'.format(epoch, itr, tloss / steps,
                                                                                      score / steps))
            tloss, score = 0, 0
        itr += 1

filename_pth = 'checkpoint_mobilenet.pth'
torch.save(model.state_dict(), filename_pth)
