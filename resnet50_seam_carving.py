import os
import torch
from PIL import Image
import torchvision
from torchvision import transforms, models as models
# import seam_carving
from carve import resize
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchinfo import summary

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model_list = [
    torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True),
    # torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True),
    # torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True),
    # torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True),
    # models.efficientnet_b1(pretrained=True),
    # models.mnasnet1_0(pretrained=True),
    # models.mobilenet_v2(pretrained=True),
    # models.resnext50_32x4d(pretrained=True),
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

        # dst = seam_carving.resize(
        dst = resize(
            image, (self.target_size[0], self.target_size[1]),
            energy_mode=self.energy_mode,  # Choose from {backward, forward}
            order='width-first',  # Choose from {width-first, height-first}
            keep_mask=None
        )

        return Image.fromarray(dst)

    def __repr__(self):
        return self.__class__.__name__ + "(target_size=%d, energe_mode=%s)" % (self.target_size[0], self.energy_mode)

input_size = 128
resize_size = int(input_size * 256 / 224)
interpolation = torchvision.transforms.InterpolationMode.BILINEAR
energy_mode = "backward"

preprocess = transforms.Compose([
    transforms.Resize(resize_size, interpolation=interpolation),
    SeamCarvingResize(resize_size, energy_mode=energy_mode),
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
    shuffle=True,
    num_workers=0,
)

with torch.no_grad():
    for model in model_list:
        # with open("%s.txt" % model._get_name(), "w") as f:
        f = open("%s_%d_%d_%s.txt" % (model._get_name(), resize_size, input_size, energy_mode), "w")
        model.eval()
        model_summary = summary(model, input_size=(1, 3, input_size, input_size))
        f.write(str(model_summary) + "\n\n")
        f.write(str(preprocess) + "\n")
        model = model.to(device)
        total, correct, correct_5 = 0, 0, 0

        # for data in tqdm(test_loader):
        for i, data in enumerate(test_loader):
            # print("%dth batch" % i)
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_top5 = torch.topk(outputs, 5)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct_5 += torch.eq(predicted_top5, labels.view(-1, 1)).sum().float().item()

            print("Top-1 Accuracy of the network on the [%d] test images: [%f]" % (total, correct / total))
            f.write("Top-1 Accuracy of the network on the [%d] test images: [%f]\n" % (total, correct / total))
            print("Top-5 Accuracy of the network on the [%d] test images: [%f]" % (total, correct_5 / total))
            f.write("Top-5 Accuracy of the network on the [%d] test images: [%f]\n" % (total, correct_5 / total))

            top_1_list = (predicted == labels).detach().cpu().numpy().astype(np.bool_)
            top_5_list = torch.eq(predicted_top5, labels.view(-1, 1)).sum(axis=1).detach().cpu().numpy().astype(np.bool_)

            ## True case plot...
            true_image_idx = np.where(top_1_list == True)[0][-1]
            true_image_num = i * test_loader.batch_size + true_image_idx
            true_image = Image.open(test_loader.dataset.imgs[true_image_num][0].replace("\\", "/")).convert("RGB")
            true_image_resized = transforms.Resize(resize_size, interpolation=interpolation)(true_image)
            true_image_sc = np.array(SeamCarvingResize(resize_size, energy_mode=energy_mode)(true_image_resized))
            true_image_label = predicted.detach().cpu().numpy()[true_image_idx]
            true_image_label_5 = predicted_top5.detach().cpu().numpy()[true_image_idx]
            true_label = dataset.classes[true_image_label][0]
            true_label_5 = np.array(dataset.classes, dtype=object)[true_image_label_5]
            true_prob = torch.nn.Softmax(dim=1)(outputs)[true_image_idx]
            prob_top5, _ = torch.topk(true_prob, 5)
            prob_top5 = prob_top5.detach().cpu().numpy().astype(np.float32)

            fig1 = plt.figure(1)
            plt.subplot(311)
            plt.imshow(true_image, aspect="equal")

            plt.subplot(312)
            plt.imshow(true_image_sc, aspect="equal")

            plt.subplot(313)
            df = pd.DataFrame(prob_top5.reshape(-1, 5), columns=[true_label_5[0][0], true_label_5[1][0], true_label_5[2][0], true_label_5[3][0], true_label_5[4][0]])
            plt.table(cellText=df.values, colLabels=df.columns, loc='center')
            plt.axis("off")

            plt.suptitle("top-1 True, top-5 True \n%s" % test_loader.dataset.imgs[true_image_num][0].replace("\\", "/"))
            plt.tight_layout()
            plt.show()
            fig1.savefig("True_%s" % test_loader.dataset.imgs[true_image_num][0].replace("\\", "/").split("/")[-1], dpi=200)

            ## Flase case plot...
            true_image_idx = np.where(top_1_list == False)[0][-1]
            true_image_num = i * test_loader.batch_size + true_image_idx
            true_image = Image.open(test_loader.dataset.imgs[true_image_num][0].replace("\\", "/")).convert("RGB")
            true_image_resized = transforms.Resize(resize_size, interpolation=interpolation)(true_image)
            true_image_sc = np.array(SeamCarvingResize(resize_size, energy_mode=energy_mode)(true_image_resized))
            true_image_label = predicted.detach().cpu().numpy()[true_image_idx]
            true_image_label_5 = predicted_top5.detach().cpu().numpy()[true_image_idx]
            true_label = dataset.classes[true_image_label][0]
            true_label_5 = np.array(dataset.classes, dtype=object)[true_image_label_5]
            true_prob = torch.nn.Softmax(dim=1)(outputs)[true_image_idx]
            prob_top5, _ = torch.topk(true_prob, 5)
            prob_top5 = prob_top5.detach().cpu().numpy().astype(np.float32)

            fig2 = plt.figure(2)
            plt.subplot(311)
            plt.imshow(true_image, aspect="equal")

            plt.subplot(312)
            plt.imshow(true_image_sc, aspect="equal")

            plt.subplot(313)
            df = pd.DataFrame(prob_top5.reshape(-1, 5), columns=[true_label_5[0][0], true_label_5[1][0], true_label_5[2][0], true_label_5[3][0], true_label_5[4][0]])
            plt.table(cellText=df.values, colLabels=df.columns, loc='center')
            plt.axis("off")

            plt.suptitle("top-1 False, top-5 %s \n%s" % (bool(torch.eq(predicted_top5, labels.view(-1, 1)).sum(axis=1)[true_image_idx].item()),
                                                      test_loader.dataset.imgs[true_image_num][0].replace("\\", "/")))
            plt.tight_layout()
            plt.show()
            fig2.savefig("False_%s" % test_loader.dataset.imgs[true_image_num][0].replace("\\", "/").split("/")[-1], dpi=200)


        f.close()


