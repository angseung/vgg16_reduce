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
from image_lists import image_list_128, image_list_160


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

# crop = transforms.CenterCrop(input_size)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

perf = {}

input_size = 128
num_true = 0

# image_list = image_list_128 if input_size == 128 else image_list_160
random.shuffle(image_list_128)
image_list = image_list_128[:1000]

for image in image_list:
    print("processing %s file..." % image[0])
    resize_size = int(input_size * (256 / 224))
    input_image_PIL = Image.open(image[0])

    if len(input_image_PIL.size) != 3:
        input_image_PIL = input_image_PIL.convert("RGB")

    # assert len(np.array(input_image_PIL).shape) == 3
    input_image = transforms.ToTensor()(input_image_PIL)
    input_image = transforms.Resize(resize_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)(input_image)
    # input_image = normalize(input_image)

    ## (ch, w, h) -> (w, h, ch)
    input_image = input_image.numpy().transpose([1, 2, 0])
    # plt.imshow(input_image)
    # plt.show()
    dy = input_size - input_image.shape[0]
    dx = input_size - input_image.shape[1]

    """
    original code here...
    """
    # resized_image = seam_carve(input_image, dy, dx).transpose([2, 0, 1])

    """
    for fast seam carving here...
    """
    input_image = cv2.normalize(input_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    resized_image = seam_carving.resize(input_image, (input_size, input_size), energy_mode="backward", order="width-first").transpose([2, 0, 1])
    resized_image = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # resized_image = seam_carve(input_image, 1, 1)
    resized_tensor = torch.from_numpy(resized_image.copy())
    resized_tensor = normalize(resized_tensor)
    # Image.fromarray(resized_image.astype(np.uint8)).show()
    # resized_image = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # resized_image = torch.from_numpy(resized_image.astype(np.float32).transpose([2, 0, 1]).copy())
    # input_image = torch.from_numpy(seam_carve(input_image, 0, 0).transpose([2, 0, 1]))

    # preprocess = transforms.Compose([
        # transforms.CenterCrop(input_size),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # input_tensor = preprocess(resized_image)
    input_batch = resized_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch.float())
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    _, predicted = torch.max(output.data, 1)
    print("with seam_carving", predicted.data.item(), ", with bilinear", image[1], ", which was %s" % image[2])

    num_true += int(predicted.data.item() == image[1])

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)

acc = num_true / len(image_list)
print(f"acc : [{acc}]")