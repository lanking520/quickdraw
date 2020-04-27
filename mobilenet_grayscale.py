import torch
import torchvision


def squeeze_weights(m):
    m.weight.data = m.weight.data.sum(dim=1)[:, None]
    m.in_channels = 1


# https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
def get_MobileNet_grayscale(classes, pretrained=True):
    model = torchvision.models.mobilenet_v2(pretrained=pretrained)
    first_layer = list(model.features.children())[0]
    model_body = list(model.features.children())[1:]

    # Apply on First Convolution layer to take 1 dim
    list(first_layer.children())[0].apply(squeeze_weights)
    model.features = torch.nn.Sequential(first_layer, *model_body)
    linear = torch.nn.Linear(1280, classes)
    torch.nn.init.normal_(linear.weight, 0, 0.01)
    torch.nn.init.zeros_(linear.bias)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        linear
    )
    return model
