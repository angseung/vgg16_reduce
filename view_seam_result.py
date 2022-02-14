import os
import numpy as np
import torch
from PIL import Image
import cv2
import seam_carving
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision import models as models

root = "C:/imagenet/val"
label_list = os.listdir(root)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
interpolation = transforms.InterpolationMode.BILINEAR
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = models.resnext50_32x4d(pretrained=True)
model = model.to(device)
model.eval()
mode = "forward"

with torch.no_grad():
    for i, label_dir in enumerate(label_list):
        img_list = os.listdir(root + "/" + label_dir)
        img_path = root + "/" + label_dir + "/" + img_list[0]
        curr_image = np.array(Image.open(img_path).convert("RGB"))
        print("processing %s file..." % img_list[0])

        fig = plt.figure()
        plt.subplot(2, 2, 1)
        plt.title("original")
        plt.imshow(np.array(curr_image))

        plt.subplot(2, 2, 2)
        # img_160 = seam_carving.resize(curr_image, (160, 160), energy_mode="backward")
        img_160 = seam_carving.resize(curr_image, (160, 160), energy_mode=mode)
        input_tensor = preprocess(Image.fromarray(img_160))
        input_batch = input_tensor.unsqueeze(0)
        output = model(input_batch.float().to("cuda"))
        _, predicted = torch.max(output.data, 1)
        plt.title("size==160, [%s]" % str(predicted.data.item() == i))
        plt.imshow(img_160)
        IS_TRUE = predicted.data.item() == i

        plt.subplot(2, 2, 3)
        img_128 = seam_carving.resize(curr_image, (128, 128), energy_mode=mode)
        input_tensor = preprocess(Image.fromarray(img_128))
        input_batch = input_tensor.unsqueeze(0)
        output = model(input_batch.float().to("cuda"))
        _, predicted = torch.max(output.data, 1)
        plt.title("size==128, [%s]" % str(predicted.data.item() == i))
        plt.imshow(img_128)

        plt.subplot(2, 2, 4)
        img_64 = seam_carving.resize(curr_image, (64, 64), energy_mode=mode)
        input_tensor = preprocess(Image.fromarray(img_64))
        input_batch = input_tensor.unsqueeze(0)
        output = model(input_batch.float().to("cuda"))
        _, predicted = torch.max(output.data, 1)
        plt.title("size==64, [%s]" % str(predicted.data.item() == i))
        plt.imshow(img_64)

        plt.tight_layout()
        plt.suptitle(label_dir)
        # plt.show()

        fig.savefig("./outs/%d_%s.png" % (IS_TRUE, label_dir), dpi=200)
