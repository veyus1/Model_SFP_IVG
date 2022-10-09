
# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import torch
import torchvision.ops
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.utils import draw_segmentation_masks
import transforms
import utils
from Model_maskRCNN_droplets import DropletsMaskDataset, get_transform, get_model_object_detection
from engine import evaluate
import pandas as pd
import os
import shutil
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms as T

"""remove faulty imgs and masks"""

c = 1
bad_list = list(pd.read_csv("bad_imgs").iloc[:,1])
imgs_list = os.listdir("DropletsMask/raw")
masks_list = os.listdir("DropletsMask/masks")
for img_name in bad_list:
        if img_name in imgs_list:
                os.remove(os.path.join("DropletsMask/raw",img_name))
                os.remove(os.path.join("DropletsMask/masks", img_name))
                print(f"{img_name} has been removed from raw and masks, total no: {c}")
                c+=1
print("ok")





r"""
#raw_path = r"C:\Users\smveerso\Desktop\IOU0,05"
path_1 = r"C:\Users\smveerso\Desktop\IOU0,05\0,2"
path_2 = r"C:\Users\smveerso\Desktop\IOU0,05\0,4"
path_3 = r"C:\Users\smveerso\Desktop\IOU0,05\0,6"
path_4 = r"C:\Users\smveerso\Desktop\IOU0,05\0,8"

#raw_folder = os.listdir(raw_path)
folder_1 = os.listdir(path_1)
folder_2 = os.listdir(path_2)
folder_3 = os.listdir(path_3)
folder_4 = os.listdir(path_4)

path_tet = r"C:\Users\smveerso\Desktop\IOU0,05"
folder_tet = os.listdir(path_tet)


for pred_image in folder_1:

    ax1 = plt.subplot(1, 5, 1)
    plt.imshow(Image.open(os.path.join(path_1, pred_image)), cmap= "gray")

    ax3 = plt.subplot(1, 5, 2, sharex=ax1, sharey=ax1)
    plt.imshow(Image.open(os.path.join(path_2, pred_image)), cmap= "gray")

    ax4 = plt.subplot(1, 5, 3, sharex=ax1, sharey=ax1)
    plt.imshow(Image.open(os.path.join(path_3, pred_image)), cmap= "gray")

    ax5 = plt.subplot(1, 5, 4, sharex=ax1, sharey=ax1)
    plt.imshow(Image.open(os.path.join(path_4, pred_image)), cmap= "gray", )

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0.35, right=0.9, wspace=0.1)
    plt.show()
"""

r"""
raw_path = r"C:\Users\veyse\OneDrive\Desktop\testing imgs\raw_vis"
path_1 = r"C:\Users\veyse\OneDrive\Desktop\testing imgs\const mask thres 0.7\0.001"
path_2 = r"C:\Users\veyse\OneDrive\Desktop\testing imgs\const mask thres 0.7\0.01"
path_3 = r"C:\Users\veyse\OneDrive\Desktop\testing imgs\const mask thres 0.7\0.05"
path_4 = r"C:\Users\veyse\OneDrive\Desktop\testing imgs\const mask thres 0.7\0.1"
path_5 = r"C:\Users\veyse\OneDrive\Desktop\testing imgs\const mask thres 0.7\0.5"

raw_folder = os.listdir(raw_path)
folder_1 = os.listdir(path_1)
folder_2 = os.listdir(path_2)
folder_3 = os.listdir(path_3)
folder_4 = os.listdir(path_4)
folder_5 = os.listdir(path_5)

for raw_image,pred_image in zip(raw_folder,folder_1):
    ax1 = plt.subplot(1,6,1)
    plt.imshow(Image.open(os.path.join(raw_path, raw_image)))

    ax2 = plt.subplot(1, 6, 2, sharex=ax1, sharey=ax1)
    plt.imshow(Image.open(os.path.join(path_1, pred_image)), cmap= "gray")

    ax3 = plt.subplot(1, 6, 3, sharex=ax1, sharey=ax1)
    plt.imshow(Image.open(os.path.join(path_2, pred_image)), cmap= "gray")

    ax4 = plt.subplot(1, 6, 4, sharex=ax1, sharey=ax1)
    plt.imshow(Image.open(os.path.join(path_3, pred_image)), cmap= "gray")

    ax5 = plt.subplot(1, 6, 5, sharex=ax1, sharey=ax1)
    plt.imshow(Image.open(os.path.join(path_4, pred_image)), cmap= "gray", )

    ax6 = plt.subplot(1, 6, 6, sharex=ax1, sharey=ax1)
    plt.imshow(Image.open(os.path.join(path_5, pred_image)), cmap="gray", )

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0.35, right=0.9, wspace=0.1)
    plt.show()"""
"""
#trafos = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])
dataset = DropletsMaskDataset("DropletsMask",transforms=get_transform(train=True))

image, target = dataset[7]
plt.subplot(1,2,1)
plt.imshow(image.permute(1,2,0))


plt.subplot(1,2,2)
a = draw_segmentation_masks(torch.zeros(3,target["masks"].size(dim=1),target["masks"].size(dim=2)).to(dtype=torch.uint8),target["masks"].to(dtype=torch.bool))
plt.imshow(a.permute(1,2,0))
plt.show()
print("ok")"""