import os
import torch
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
else:
    device = "cpu"

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

model_list = [
    torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True),
    # torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True),
    # torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True),
    # torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True),
    # models.efficientnet_b1(pretrained=True),
    # models.mnasnet1_0(pretrained=True),
    # models.mobilenet_v2(pretrained=True),
    # models.resnext50_32x4d(pretrained=True),
]

input_size = 64
resize_size = int(input_size * 256 / 224)
interpolation = torchvision.transforms.InterpolationMode.BILINEAR
energy_mode = "backward"
mode = "fine-tune"  # ["fine-tune", "feature-extract"]

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
train_dataset = ImageFolder(
    root="C:/imagenet/train_%d" % input_size, transform=preprocess
)
trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
)

test_dataset = ImageFolder(root="C:/imagenet/val_%d" % input_size, transform=preprocess)
testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

model = model_list[0]
if mode == "feature-extract":
    for params in model.parameters():
        params.requires_grad = False

# model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)

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
                loss=train_loss / (batch_idx + 1), accuracy=100.0 * correct / total
            )

    with open("outputs/" + dir_path + "/log.txt", "a") as f:
        f.write(
            "Epoch [%d] |Train| Loss: %.3f, Acc: %.3f \t"
            % (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total)
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
                    loss=test_loss / (batch_idx + 1), accuracy=100.0 * correct / total
                )
    acc = 100.0 * correct / total

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
        f.write("|Test| Loss: %.3f, Acc: %.3f \n" % (test_loss / (batch_idx + 1), acc))

    # return (epoch, test_loss, acc)


start_epoch = 0
max_epoch = 50

for model in model_list:
    netkey = model._get_name()
    log_path = "outputs/" + netkey
    model = model.to(device)

    os.makedirs(log_path, exist_ok=True)

    with open(log_path + "/log.txt", "w") as f:
        f.write("Networks : %s\n" % netkey)
        m_info = summary(model, (1, 3, input_size, input_size), verbose=0)
        f.write("%s\n" % str(m_info))

    # optimizer = torch.optim.SGD(model.conv1.parameters(), lr = 0.001, momentum=0.9)
    ## original loss and optim config
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    ##########################
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=int(max_epoch * 1.0)
    # )

    # if config["train_resume"]:
    #     # Load checkpoint.
    #     print("==> Resuming from checkpoint..")
    #     # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load(log_path + "/ckpt.pth")
    #     model.load_state_dict(checkpoint["net"])
    #     scheduler.load_state_dict(checkpoint["scheduler"])
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     best_acc = checkpoint["acc"]
    #     start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, max_epoch):
        train(epoch, netkey)
        test(epoch, netkey)
        # scheduler.step()
