import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from mobilenet_grayscale import get_MobileNet_grayscale

DATA_DIR = "input/images"

BATCHSIZE = 256
STEPS = 1000
EPOCHS = 16
NCATS = 340
DEVICE = "cpu"
torch.manual_seed(1987)

def writeToFile(text):
    file_name = "output.log"
    with open(file_name, "a+") as f:
        f.write(text + "\n")

image_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor()
])

total_data = datasets.ImageFolder(DATA_DIR, transform=image_transform)

train_length = int(len(total_data) * 0.7)
valid_length = int(len(total_data) * 0.2)
test_length = len(total_data) - train_length - valid_length

train_data, valid_data, test_data = random_split(total_data, [train_length, valid_length, test_length])

train_loader = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True)
valid_loader = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True)
test_loader = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True)

writeToFile("Trainset {} ValidSet{} testSet {}".format(train_length, test_length, valid_length))

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


model = get_MobileNet_grayscale(NCATS, pretrained=False)
if os.path.exists('checkpoint_mobilenet.pth'):
    writeToFile("Found checkpoint file, continuing training...")
    model.load_state_dict(torch.load('checkpoint_mobilenet.pth'))
if DEVICE is "cuda":
    model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = torch.nn.CrossEntropyLoss()

lsize = len(train_loader)
itr = 1
model.train()
tloss, score = 0, 0
for epoch in range(EPOCHS):
    writeToFile("epoch " + str(epoch) + " start")
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        tloss += loss.item()
        score += accuracy(output, y)[0].item()
        if itr % STEPS == 0:
            writeToFile('Epoch {} Iteration {} -> Train Loss: {:.4f}, MAP@3: {:.3f}'.format(epoch, itr, tloss / STEPS,
                                                                                      score / STEPS))
            tloss, score = 0, 0
        itr += 1
    vloss, vscore = validation(model, valid_loader, DEVICE, criterion, accuracy)
    writeToFile("Epoch {} -> Validation Loss: {:.4f}, Map@3: {:.3f}".format(epoch, vloss, vscore))
    filename_pth = 'checkpoint' + str(epoch) + '_mobilenet.pth'
    torch.save(model.state_dict(), filename_pth)
