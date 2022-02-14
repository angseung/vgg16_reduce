import random
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
from image_lists import image_list_128, image_list_160, image_list_196


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


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


preprocess = transforms.Compose(
    [
        # transforms.Resize(256),
        SeamCarvingResize(256, energy_mode="forward"),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = models.resnext50_32x4d(pretrained=True)
model = model.to(device)
model.eval()

interpolation_list = [
    torchvision.transforms.InterpolationMode.BILINEAR,
    torchvision.transforms.InterpolationMode.NEAREST,
    torchvision.transforms.InterpolationMode.BICUBIC,
]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

perf = {}

input_size = 196
num_true = 0

if input_size == 128:
    image_list = image_list_128
elif input_size == 160:
    image_list = image_list_160
elif input_size == 196:
    image_list = image_list_196
else:
    raise NotImplementedError

random.shuffle(image_list)

for image in image_list[:1000]:
    print("processing %s file..." % image[0])
    resize_size = int(input_size * (256 / 224))
    input_image_PIL = Image.open(image[0])

    if len(input_image_PIL.size) != 3:
        input_image_PIL = input_image_PIL.convert("RGB")

    input_image = transforms.ToTensor()(input_image_PIL)
    input_image = transforms.Resize(
        resize_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )(input_image)

    ## (ch, w, h) -> (w, h, ch)
    input_image = input_image.numpy().transpose([1, 2, 0])
    dy = input_size - input_image.shape[0]
    dx = input_size - input_image.shape[1]

    """
    original code here...
    """
    # resized_image = seam_carve(input_image, dy, dx).transpose([2, 0, 1])

    """
    for fast seam carving here...
    """
    input_image = cv2.normalize(
        input_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    resized_image = seam_carving.resize(
        input_image,
        (input_size, input_size),
        energy_mode="backward",
        order="width-first",
    ).transpose([2, 0, 1])
    resized_image = cv2.normalize(
        resized_image,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    # resized_image = seam_carve(input_image, 1, 1)
    resized_tensor = torch.from_numpy(resized_image.copy())
    resized_tensor = normalize(resized_tensor)
    input_batch = resized_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model(input_batch.float())
    _, predicted = torch.max(output.data, 1)
    print(
        "with seam_carving",
        predicted.data.item(),
        ", with bilinear",
        image[1],
        ", which was %s" % image[2],
    )

    num_true += int(predicted.data.item() == image[1])

acc = num_true / 1000
print(f"acc : [{acc}]")
