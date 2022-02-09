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

# crop = transforms.CenterCrop(input_size)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

perf = {}

image_list = [
['C:/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG', 0, False],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00002138.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00003014.JPEG', 0, False],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00006697.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00007197.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00009111.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00009191.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00009346.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00009379.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00009396.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00010306.JPEG', 0, False],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00011233.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00011993.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00012503.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00013716.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00016018.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00017472.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00017699.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00017700.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00017995.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00018317.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00021740.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00023559.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00024235.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00024327.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00025129.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00025527.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00026064.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00026397.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00028158.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00029930.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00030740.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00031094.JPEG', 0, False],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00031333.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00034654.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00037375.JPEG', 0, False],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00037383.JPEG', 0, False],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00037596.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00037834.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00037861.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00039905.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00040358.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00040833.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00041939.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00045866.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00045880.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00046252.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00046499.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00048204.JPEG', 0, True],
['C:/imagenet/val/n01440764/ILSVRC2012_val_00048969.JPEG', 0, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00000236.JPEG', 1, False],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00000262.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00000307.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00000994.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00002241.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00002848.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00003150.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00003735.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00004655.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00004677.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00005870.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00006007.JPEG', 1, False],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00006216.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00009034.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00010363.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00010509.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00011914.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00012880.JPEG', 1, False],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00013513.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00013623.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00016962.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00018075.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00019459.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00020436.JPEG', 1, False],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00020785.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00020822.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00021905.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00022138.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00023869.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00028713.JPEG', 1, False],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00030060.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00030217.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00031138.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00032235.JPEG', 1, False],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00032258.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00032675.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00034386.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00037846.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00038057.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00044095.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00045761.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00046915.JPEG', 1, False],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00046969.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00047396.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00047561.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00048840.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00048864.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00049585.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00049617.JPEG', 1, True],
['C:/imagenet/val/n01443537/ILSVRC2012_val_00049712.JPEG', 1, True],
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
