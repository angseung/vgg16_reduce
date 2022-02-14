import numpy as np
import cv2
import torch
import torchvision
from torchvision import models as models
from PIL import Image
from torchvision import datasets, transforms
from torchinfo import summary
from matplotlib import pyplot as plt
from tqdm import tqdm
import seam_carving
from seam_carving_v1 import seam_carve
from utils import get_flatten_model, get_trimed_model, channel_repeat


class SeamCarvingResize:
    def __init__(self, size, energy_mode="backward"):
        if isinstance(size, int):
            self.target_size = (size, size)
        elif isinstance(size, tuple):
            if len(size) == 2:
                self.target_size = size
            else:
                raise ValueError("Size must be int or tuple with length 2 (h, w)")

        self.energy_mode = energy_mode  # 'forward' or 'backward'

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            image = np.array(img)

        dst = seam_carving.resize(
            image,
            (self.target_size[0], self.target_size[1]),
            energy_mode=self.energy_mode,  # Choose from {backward, forward}
            order="width-first",  # Choose from {width-first, height-first}
            keep_mask=None,
        )

        return Image.fromarray(dst)


input_size = 128
interpolation = torchvision.transforms.InterpolationMode.BILINEAR
preprocess = transforms.Compose(
    [
        transforms.Resize(256, interpolation=interpolation),
        SeamCarvingResize(256, energy_mode="forward"),
        transforms.CenterCrop(input_size),
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

interpolation_list = [
    torchvision.transforms.InterpolationMode.BILINEAR,
]
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

perf = {}

image_list = [
    ["E:/imagenet-mini/val/n01440764/ILSVRC2012_val_00009111.JPEG", 0, False],
    ["E:/imagenet-mini/val/n01440764/ILSVRC2012_val_00030740.JPEG", 0, True],
    ["E:/imagenet-mini/val/n01440764/ILSVRC2012_val_00046252.JPEG", 0, False],
]

input_size = 160

for image in image_list:
    print("processing %s file..." % image[0])
    resize_size = int(input_size * (256 / 224))
    input_image = np.array(Image.open(image[0]))
    # plt.imshow(input_image)
    # plt.show()
    dy = resize_size - input_image.shape[0]
    dx = resize_size - input_image.shape[1]
    # resized_image = seam_carving.resize(input_image, (resize_size, resize_size), energy_mode="backward", order="width-first")
    # resized_image = seam_carving.resize(input_image, (resize_size, resize_size), energy_mode="backward", order="height-first")
    # resized_image = seam_carving.resize(input_image, (resize_size, resize_size), energy_mode="forward", order="width-first")
    # resized_image = seam_carving.resize(input_image, (resize_size, resize_size), energy_mode="forward", order="height-first")
    resized_image = seam_carve(input_image, dy, dx)
    Image.fromarray(resized_image.astype(np.uint8)).show()
    resized_image = cv2.normalize(
        resized_image,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    resized_image = torch.from_numpy(
        resized_image.astype(np.float32).transpose([2, 0, 1]).copy()
    )
    # input_image = torch.from_numpy(seam_carve(input_image, 0, 0).transpose([2, 0, 1]))

    preprocess = transforms.Compose(
        [
            transforms.CenterCrop(input_size),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(resized_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model(input_batch.float())
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    _, predicted = torch.max(output.data, 1)
    print(
        "with seam_carving",
        predicted,
        ", with bilinear",
        image[1],
        ", which was %s" % image[2],
    )
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
