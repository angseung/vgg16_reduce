import numpy as np
import torch
import torchvision
from torchvision import models as models
from torchvision import datasets, transforms as T
from torchinfo import summary
from matplotlib import pyplot as plt
from tqdm import tqdm
import seam_carving
from seam_carving_v1 import seam_carve
from utils import get_flatten_model, get_trimed_model, channel_repeat


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = models.resnext50_32x4d(pretrained=True)
model = model.to(device)
model.eval()

# input_size = 64

# input = (1, 3, input_size, input_size)
# summary(model, input)

interpolation_list = [
    torchvision.transforms.InterpolationMode.BILINEAR,
    # torchvision.transforms.InterpolationMode.NEAREST,
    # torchvision.transforms.InterpolationMode.BICUBIC,
]
# input_size_list = [224, 196, 160, 128, 64]
input_size_list = [224]

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

perf = {}

for inter in interpolation_list:
    for input_size in input_size_list:
        resize_size = int(input_size * (256 / 224))
        transform = T.Compose(
            [
                T.Resize(resize_size, interpolation=inter),
                T.CenterCrop(input_size),
                T.ToTensor(),
                normalize,
            ]
        )

        my_dataset = torchvision.datasets.ImageNet(
            root="C:/imagenet/", split="val", transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            my_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )
        correct = 0
        total = 0

        true_or_false = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                # print(f"processing {i}th batch...")
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # (sample_fname, l) = test_loader.dataset.samples[i]
                true_or_false += list((predicted == labels).detach().cpu().numpy())

        print(
            f"Accuracy of the network on the {total} test images: {100 * correct / total} %"
        )
        perf[100 * correct / total] = [inter, input_size]

img_info = test_loader.dataset.imgs

with open("true_%d.txt" % input_size, "w") as f:
    for i in range(len(true_or_false)):
        f.write(
            "['%s', %d, %s],\n"
            % (img_info[i][0].replace("\\", "/"), img_info[i][1], str(true_or_false[i]))
        )
