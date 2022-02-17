import os
os.environ["CUDA_VISIBLE_DEVICE"] = "0"
import torch
from torch import nn as nn
from PIL import Image
import torchvision
from torchvision import transforms, models as models
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchinfo import summary
from torchvision.datasets import ImageFolder
from utils import SeamCarvingResize, Resize
from carve import resize

if torch.cuda.is_available():
    device = "cuda"
    # device = torch.device("cuda:0")
else:
    device = "cpu"

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
print(torch.cuda.current_device())

model = models.resnet18(pretrained=True)

best_acc = 0
input_size = 64
resize_size = int(input_size * 256 / 224)
interpolation = torchvision.transforms.InterpolationMode.BILINEAR
energy_mode = "backward"
mode = "fine-tune"  # ["fine-tune", "feature-extract"]

preprocess_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

preprocess_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
train_dataset = ImageFolder(
    root="C:/tiny-imagenet-200/train", transform=preprocess_train
)
trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
)

test_dataset = ImageFolder(root="C:/tiny-imagenet-200/val_torch", transform=preprocess_test)
testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)

if mode == "feature-extract":
    for params in model.parameters():
        params.requires_grad = False
elif mode == "fine-tune":
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 200)

# model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.fc = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)

# Training
def train(epoch, dir_path=None, plotter=None) -> None:
    print("\nEpoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct = 0
    correct_5 = 0
    total = 0

    with tqdm(trainloader, unit="batch") as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            tepoch.set_description(f"Train Epoch {epoch}")

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            _, predicted_top5 = torch.topk(outputs, 5)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            correct_5 += (
                torch.eq(predicted_top5, targets.view(-1, 1)).sum().float().item()
            )

            tepoch.set_postfix(
                loss=train_loss / (batch_idx + 1), accuracy=100.0 * correct / total,
                top_5=100.0 * correct_5 / total
            )

    with open("outputs/" + dir_path + "/log.txt", "a") as f:
        f.write(
            "Epoch [%d] |Train| Loss: %.3f, Acc: %.3f %.3f\t"
            % (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total,
               100.0 * correct_5 / total)
        )

    # return (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total)


def test(epoch, dir_path=None, plotter=None) -> None:
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    correct_5 = 0
    total = 0

    if dir_path is None:
        dir_path = "outputs/checkpoint"
    else:
        dir_path = "outputs/" + dir_path

    with torch.no_grad():
        with tqdm(testloader, unit="batch") as tepoch:
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                tepoch.set_description(f"Test Epoch {epoch}")

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                _, predicted_top5 = torch.topk(outputs, 5)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                correct_5 += (
                    torch.eq(predicted_top5, targets.view(-1, 1)).sum().float().item()
                )

                tepoch.set_postfix(
                    loss=test_loss / (batch_idx + 1), accuracy=100.0 * correct / total,
                top_5=100.0 * correct_5 / total
                )
    acc = 100.0 * correct / total
    acc5 = 100.0 * correct_5 / total

    # Save checkpoint.
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save(state, "./" + dir_path + "/ckpt.pth")

        best_acc = acc

    with open(dir_path + "/log.txt", "a") as f:
        f.write("|Test| Loss: %.3f, Acc: %.3f %.3f\n" % (test_loss / (batch_idx + 1), acc, acc5))

    # return (epoch, test_loss, acc)


start_epoch = 0
max_epoch = 50

netkey = model._get_name()
log_path = "outputs/" + netkey
model = model.to(device)

os.makedirs(log_path, exist_ok=True)

with open(log_path + "/log.txt", "w") as f:
    f.write("Networks : %s\n" % netkey)
    m_info = summary(model, (1, 3, input_size, input_size), verbose=0)
    f.write("%s\n" % str(m_info))

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

for epoch in range(start_epoch, max_epoch):
    train(epoch, netkey)
    test(epoch, netkey)
    scheduler.step()
