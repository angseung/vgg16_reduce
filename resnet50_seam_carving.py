import os
import torch
from PIL import Image
import torchvision
from torchvision import transforms, models as models
import seam_carving
import numpy as np
from tqdm import tqdm
from torchinfo import summary

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model_list = [
    torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True),
    torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True),
    torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True),
    torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True),
    # models.efficientnet_b1(pretrained=True),
    models.mnasnet1_0(pretrained=True),
    models.mobilenet_v2(pretrained=True),
    models.resnext50_32x4d(pretrained=True),
]

class SeamCarvingResize:
    def __init__(self, size, energy_mode='backward'):
        if isinstance(size, int):
            self.target_size = (size, size)
        elif isinstance(size, tuple):
            if len(size) == 2:
                self.target_size = size
            else:
                raise ValueError('Size must be int or tuple with length 2 (h, w)')

        self.energy_mode = energy_mode  # 'forward' or 'backward'

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            image = np.array(img)

        dst = seam_carving.resize(
            image, (self.target_size[0], self.target_size[1]),
            energy_mode=self.energy_mode,  # Choose from {backward, forward}
            order='width-first',  # Choose from {width-first, height-first}
            keep_mask=None
        )

        return Image.fromarray(dst)

input_size = 128
interpolation = torchvision.transforms.InterpolationMode.BILINEAR
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=interpolation),
    SeamCarvingResize(256, energy_mode='forward'),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = torchvision.datasets.ImageNet(
    root="C:/imagenet/", split="val", transform=preprocess
)
test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

with torch.no_grad():
    for model in model_list:
        model.eval()
        summary(model, input_size=(1, 3, input_size, input_size))
        model = model.to(device)
        total, correct, correct_5 = 0, 0, 0

        # for data in tqdm(test_loader):
        #     images, labels = data
        #     images, labels = images.to(device), labels.to(device)
        #     outputs = model(images)
        #     _, predicted = torch.max(outputs.data, 1)
        #     _, predicted_top5 = torch.topk(outputs, 5)
        #
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum().item()
        #     correct_5 = torch.eq(predicted_top5, labels.view(-1, 1)).sum().float().item()
