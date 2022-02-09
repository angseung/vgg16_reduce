import numpy as np
import torch
import torchvision
from torchvision import models as models
from torchvision import datasets, transforms as T
from torchinfo import summary
from matplotlib import pyplot as plt
from utils import get_flatten_model, get_trimed_model, channel_repeat


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = models.vgg16(pretrained=True)
model = model.to(device)
model.eval()

f_model = get_flatten_model(model)
new_model = torch.nn.Sequential(*f_model).to(device)
new_model.eval()

input_size = 224
resize_size = 112
interpolation = torchvision.transforms.InterpolationMode.BILINEAR
# interpolation = torchvision.transforms.InterpolationMode.NEAREST
# interpolation = torchvision.transforms.InterpolationMode.BICUBIC
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose(
    [
        T.Resize(resize_size, interpolation=interpolation),
        T.ToPILImage(),
        channel_repeat,
        # T.CenterCrop(input_size),
        T.ToTensor(),
        normalize,
    ]
)
# my_dataset = torchvision.datasets.ImageFolder("C:/imagenet/val", transform=transform)
my_dataset = torchvision.datasets.ImageNet(
    root="C:/imagenet/", split="val", transform=transform
)
test_loader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

# remove stage 1 conv blocks...
r_model = get_trimed_model(model, start_cut=0, end_cut=4)

correct = 0
total = 0

IM_SHOW_OPT = False

with torch.no_grad():
    for i, data in enumerate(test_loader):
        print(f"processing {i}th batch...")
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # test_io(new_model, images)
        outputs = r_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if IM_SHOW_OPT:
            fig = plt.figure()
            plt.imshow(images[0, :, :, :].detach().cpu().numpy().transpose([1, 2, 0]))
            plt.show()

            fig2 = plt.figure()

            for j in range(outputs.shape[1]):
                plt.subplot(8, 8, j + 1)
                plt.imshow(outputs[0, j, :, :].detach().cpu().numpy())
                plt.show()


print(f"Accuracy of the network on the {total} test images: {100 * correct / total} %")
