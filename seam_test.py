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
input_size = 160

crop = transforms.CenterCrop(input_size)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

perf = {}

image_list = [
['C:/imagenet/val/n01560419/ILSVRC2012_val_00006603.JPEG', 16, True],
['C:/imagenet/val/n01560419/ILSVRC2012_val_00007263.JPEG', 16, False],
['C:/imagenet/val/n01560419/ILSVRC2012_val_00009807.JPEG', 16, True],
['C:/imagenet/val/n01560419/ILSVRC2012_val_00011267.JPEG', 16, False],
['C:/imagenet/val/n01582220/ILSVRC2012_val_00041051.JPEG', 18, False],
['C:/imagenet/val/n01582220/ILSVRC2012_val_00041135.JPEG', 18, False],
['C:/imagenet/val/n01582220/ILSVRC2012_val_00041206.JPEG', 18, True],
['C:/imagenet/val/n01582220/ILSVRC2012_val_00044192.JPEG', 18, True],
['C:/imagenet/val/n01582220/ILSVRC2012_val_00044507.JPEG', 18, True],
['C:/imagenet/val/n01582220/ILSVRC2012_val_00045948.JPEG', 18, False],
['C:/imagenet/val/n01582220/ILSVRC2012_val_00046917.JPEG', 18, True],
['C:/imagenet/val/n01644373/ILSVRC2012_val_00043303.JPEG', 31, True],
['C:/imagenet/val/n01644373/ILSVRC2012_val_00043369.JPEG', 31, True],
['C:/imagenet/val/n01644373/ILSVRC2012_val_00043860.JPEG', 31, True],
['C:/imagenet/val/n01644373/ILSVRC2012_val_00044855.JPEG', 31, False],
['C:/imagenet/val/n01644373/ILSVRC2012_val_00048482.JPEG', 31, False],
['C:/imagenet/val/n01644373/ILSVRC2012_val_00048995.JPEG', 31, True],
['C:/imagenet/val/n01644900/ILSVRC2012_val_00000037.JPEG', 32, False],
['C:/imagenet/val/n01644900/ILSVRC2012_val_00001938.JPEG', 32, False],
['C:/imagenet/val/n01644900/ILSVRC2012_val_00004427.JPEG', 32, False],
['C:/imagenet/val/n01644900/ILSVRC2012_val_00004724.JPEG', 32, False],
['C:/imagenet/val/n01644900/ILSVRC2012_val_00005450.JPEG', 32, False],
['C:/imagenet/val/n01644900/ILSVRC2012_val_00005532.JPEG', 32, False],
    ]

for image in image_list:
    print("processing %s file..." % image[0])
    resize_size = int(input_size * (256 / 224))
    input_image = np.array(Image.open(image[0]))
    # plt.imshow(input_image)
    # plt.show()
    dy = resize_size - input_image.shape[0]
    dx = resize_size - input_image.shape[1]
    # resized_image = seam_carving.resize(input_image, (resize_size, resize_size), energy_mode="backward", order="width-first")
    resized_image = seam_carving.resize(input_image, (resize_size, resize_size), energy_mode="backward", order="height-first")
    # resized_image = seam_carving.resize(input_image, (resize_size, resize_size), energy_mode="forward", order="width-first")
    # resized_image = seam_carving.resize(input_image, (resize_size, resize_size), energy_mode="forward", order="height-first")
    # resized_image = seam_carve(input_image, dy, dx)
    # Image.fromarray(resized_image).show()
    resized_image = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    resized_image = torch.from_numpy(resized_image.astype(np.float32).transpose([2, 0, 1]).copy())
    # input_image = torch.from_numpy(seam_carve(input_image, 0, 0).transpose([2, 0, 1]))

    preprocess = transforms.Compose([
        transforms.CenterCrop(input_size),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(resized_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch.float())
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    _, predicted = torch.max(output.data, 1)
    print("with seam_carving", predicted, ", with bilinear", image[1], ", which was %s" % image[2])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
