import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from utils import SeamCarvingResize, Resize

imagenet_path = "C:/imagenet/"
input_size = 128
energy_mode = "backward"

resize_size = int(input_size * 256 / 224)

preprocess = transforms.Compose(
    [
        Resize(resize_size, aspect="wide"),
        SeamCarvingResize(resize_size, energy_mode=energy_mode),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ]
)

dataset = torchvision.datasets.ImageNet(
    root=imagenet_path, split="val", transform=preprocess
)
test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

label_map = dataset.samples
label_dict = {}

with open("C:/imagenet/labelmap.txt", "w") as f:
    for info in label_map:
        _info = info[0].replace("\\", "/").split("/")
        label_dict[_info[3]] = info[1]

    for dir, label in label_dict.items():
        f.write("%s : %d\n" % (dir, label))

with open("C:/imagenet/labelmap.txt", "r") as f:
    new_dict = {}
    labelmaps = f.readlines()

    for str in labelmaps:
        curr_label = str.split(":")
        new_dict[curr_label[0]] = int(curr_label[1][:-1])
