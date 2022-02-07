import torch
import torchvision
from torchvision import models as models
from torchvision import datasets, transforms as T


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# model = models.vgg16(pretrained=True)
# torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model = model.to(device)
model.eval()

input_size = 224
# interpolation = torchvision.transforms.InterpolationMode.BILINEAR
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = T.Compose([T.Resize(256), T.CenterCrop(input_size), T.ToTensor(), normalize])
# my_dataset = torchvision.datasets.ImageFolder("C:/imagenet/val", transform=transform)
my_dataset = torchvision.datasets.ImageNet(
        root="C:/imagenet/", split="val", transform=transform
    )
train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=64, shuffle=True,
                               num_workers=0,)

correct = 0
total = 0

with torch.no_grad():
    for i, data in enumerate(train_loader):
        print(f"processing {i}th batch...")
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct / total} %')
