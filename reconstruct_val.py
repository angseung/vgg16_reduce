import os
import shutil

base_dir = "C:/tiny-imagenet-200/val_torch"
with open("C:/tiny-imagenet-200/val_torch/val_annotations.txt", "r") as f:
    ground_truth = f.readlines()

for curr in ground_truth:
    fname, label, _, _, _, _ = curr.split("\t")
    os.walk(base_dir)
    if not os.path.isdir(base_dir + "/" + label):
        os.mkdir(base_dir + "/" + label)

    shutil.copy(base_dir + "/images/" + fname, base_dir + "/" + label + "/" + fname)
