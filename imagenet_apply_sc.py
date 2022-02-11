import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from utils import SeamCarvingResize, Resize

imagenet_path = "C:/imagenet/"
input_size_list = [64, 128, 160, 192, 224]
interpolation = torchvision.transforms.InterpolationMode.BILINEAR
energy_mode = "backward"

for input_size in input_size_list:
    resize_size = int(input_size * 256 / 224)

    preprocess = transforms.Compose(
        [
            Resize(resize_size, aspect='wide'),
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

    ## MAKE DIRECTORIES...
    print("making val_%d directory..." % input_size)
    for path in dataset.imgs:
        fname = path[0].replace("\\", "/").split("/")
        fname[2] = fname[2] + "_%d" % input_size
        curr_dir_path = "/".join(fname[:-1])
        os.makedirs(curr_dir_path, exist_ok=True)

    ## WRITE CONFIG LOG...
    with open("C:/imagenet/val_%d/config.txt" % input_size, "w") as f:
        f.write("%s" % str(preprocess))

    for i, data in enumerate(test_loader):
        images, _ = data
        images = transforms.ToPILImage()(images.squeeze(0))
        fname = dataset.imgs[i][0].replace("\\", "/").split("/")
        fname[2] = fname[2] + "_%d" % input_size
        curr_dir_path = "/".join(fname)
        print(curr_dir_path)
        images.save(curr_dir_path)
