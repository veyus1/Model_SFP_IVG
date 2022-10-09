import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import os
# import json
import cv2
import pandas as pd
#import xlwt

import torchvision.transforms.functional as f
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import shutil
from engine import train_one_epoch, evaluate
import utils

import torchvision.transforms as T


def plot_transformed_masks(img, mask, img_tr, mask_tr):
    ax1 = plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("original raw")

    plt.subplot(1, 4, 2)
    plt.imshow(mask)
    plt.title("original mask")

    plt.subplot(1, 4, 3)
    plt.imshow(img_tr)
    plt.title("transformed raw")

    plt.subplot(1, 4, 4)
    plt.imshow(mask_tr)
    plt.title("transformed mask")
    plt.show()


# enter transformations of data and a prefix of what has been done (str):
transformations = T.Compose([T.RandomVerticalFlip(1), T.RandomHorizontalFlip(1)])
prefix = "flip"

# enter path to root folder (original imgs/masks):
#root = "DropletsMask_transformed/before"
root = "DropletsMask"

# load all image files, sorting them to
# ensure that they are aligned
imgs = list(sorted(os.listdir(os.path.join(root, "raw")), key=len))
masks = list(sorted(os.listdir(os.path.join(root, "masks")), key=len))



# set num_images (entire folder or enter explicitly)
num_imgs = len(imgs)
#num_imgs = 5

# set if you want to plot every image before saving and params for transformation
# (add new ones from torchvision functional)
plot = False
prob_threshold = 1
hor = True
ver = True
bri = True
"""torch.manual_seed(0)
np.random.seed(0)"""

PIL_to_Tensor = T.PILToTensor()
for idx in range(num_imgs):
    img_path = os.path.join(root, "raw", imgs[idx])
    mask_path = os.path.join(root, "masks", masks[idx])


    # this is to copy single droplets from one image and paste them onto another (work in progress, not finished :))
    #r = np.random.randint(low=0, high=num_imgs)
    #example_drop_raw = Image.open(os.path.join(root, "raw", imgs[0])).crop((50,50,100,100))
    #example_drop_mask = Image.open(os.path.join(root, "masks", imgs[0])).crop((50,50,100,100))
    #img_tr.paste(example_drop_raw)
    #mask_tr.paste(example_drop_mask)

    img = Image.open(img_path)
    mask = Image.open(mask_path)
    name = str(prefix + "_" + imgs[idx])

    img_tr = img.copy()
    mask_tr = mask.copy()

    if prob_threshold > np.random.random():
        #if true, perform specified augmentations here
        if hor:
            img_tr = f.hflip(img_tr)
            mask_tr = f.hflip(mask_tr)

        if ver:
            img_tr = f.vflip(img_tr)
            mask_tr = f.vflip(mask_tr)

        #if bri:


    #img_tr = transformations(img)
    #mask_tr = transformations(mask)

    img_tr.save(os.path.join("DropletsMask_transformed/after/raw",name))
    mask_tr.save(os.path.join("DropletsMask_transformed/after/masks",name))

    if plot:
        plot_transformed_masks(img,mask,img_tr,mask_tr)
