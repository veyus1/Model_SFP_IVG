# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.ops
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import utils
from Model_maskRCNN_droplets import DropletsMaskDataset, get_transform, get_model_object_detection
from Model_maskRCNN_different_backbone import get_model_w_new_bb
from engine import evaluate
import cv2
import matplotlib
# NUR WÄHREND DEM TESTEN: random seed setzen
"""torch.manual_seed(101)
np.random.seed(101)"""

"""Visualize raw image w. gt_bboxes, gt_masks and prediction masks next to each other, takes data from DropletsMasks"""

dataset = DropletsMaskDataset('DropletsMask', get_transform(train=False))
#indices = list(range(len(dataset)))
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:])
#dataset = dataset_test
data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

font = {'family': 'Times New Roman',
        'size': 15}

matplotlib.rc('font', **font)

count = 0
# For inference
with torch.no_grad():

        #load model and weights, set to eval, change function depending on model
        #model = get_model_w_new_bb(num_classes=2)  # this for resnet18
        model = get_model_object_detection(num_classes=2) # this for resnet50
        model.load_state_dict(torch.load(r"Weights/weights_res50_900_50_100_2_cleaned_transf.pth"))
        model.to("cpu")
        model.eval()
        bad_imgs = []
        for image, targets in data_loader:
                id = targets[0]["image_id"].item()
                output = model(image)
                #mask_pred = output[0]["masks"].view(-1,output[0]["masks"].size(dim=2),output[0]["masks"].size(dim=3)).bool()

                # get masks and their scores, if they are good enough append to passed_masks list
                masks = output[0]["masks"].data
                scores = output[0]["scores"]
                boxes = output[0]["boxes"].data



                # Set threshold for mask values and objecness score of bboxes
                threshold = 0.6
                #objectness_threshold = 0.1

                passed_masks = []
                mask_pred_pass = torch.tensor((len(passed_masks), output[0]["masks"].size(dim=2), output[0]["masks"].size(dim=3)))

                # discard boxes with objectness score under threshold
                """obj_boxes = []
                obj_scores = []
                for box, score in zip(p_boxes,scores):
                        if score > objectness_threshold:
                                obj_boxes.append(box)
                                obj_scores.append(score)

                num_obj_boxes = len(obj_boxes)
                boxes = torch.stack(obj_boxes, dim = 0)
                scores = torch.as_tensor(obj_scores)"""

                # use non maximum supression (nms) to find boxes/masks with best iou:
                nms_boxes = []
                nms_masks = []
                nms_indices = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold= 0.05)
                for i in nms_indices:
                        nms_boxes.append(boxes[i])
                        nms_masks.append(masks[i])

                if len(nms_boxes) > 0:
                        boxes = torch.cat(nms_boxes).view(nms_indices.size(dim=0), 4)
                        masks = torch.cat(nms_masks).view(nms_indices.size(dim=0), 1, masks.size(dim=2),
                                                                                        masks.size(dim=3))

                for mask, score in zip(masks, scores):
                        mask[mask < threshold] = 0
                        passed_masks.append(mask)

                # create ground truth masks and bbox tensors, and a black background tensor to draw them on
                gt_boxes = targets[0]["boxes"]
                gt_masks = targets[0]["masks"].bool()
                masks_background = torch.zeros((3,targets[0]["masks"].size(dim=1),targets[0]["masks"].size(dim=2) ), dtype= torch.uint8)
                num_bbox_gt = targets[0]["boxes"].size(dim=0)
                num_segm_gt = targets[0]["masks"].size(dim=0)

                # use torchvision io functions to draw masks and bboxes, either on background or original image
                col_seg = []
                for c in range(len(passed_masks)):
                        col_seg.append((255-c, 255-c,255-c))

                col_gt = []
                for c in range(num_segm_gt):
                        col_gt.append((255 - c, 255 - c, 255 - c))

                gt_segm_masks = draw_segmentation_masks(masks_background, gt_masks, colors=col_gt)
                gt_mask_bboxes = draw_bounding_boxes(torch.mul(image[0], 255).to(dtype=torch.uint8), gt_boxes, width=1, colors=(255,0,0))

                # comment following line out, to remove pred bboxes (green)
                #gt_mask_bboxes = draw_bounding_boxes(gt_mask_bboxes, boxes, width=1,colors=(0, 255, 0))

                # plot the images
                plt.subplots(1, 3, figsize=(15,15))

                ax1 = plt.subplot(1,3,1)
                img_name = dataset.dataset.imgs[id]
                plt.imshow(gt_mask_bboxes.permute(1,2,0))
                #plt.title(f"original w gt bbox for\n {img_name}")
                plt.xlabel("x / px", size=16)
                plt.ylabel("y / px", size=16)
                plt.tick_params(labelsize=16)

                print(f"for image {img_name}: gt number of bboxes: {num_bbox_gt}; gt number of segm masks: {num_segm_gt}; number of pred masks: {len(passed_masks)}; id: {id}")

                ax2 = plt.subplot(1,3,2, sharex=ax1, sharey=ax1)
                plt.imshow(gt_segm_masks.permute(1,2,0))
                #plt.title("gt segmentation masks")
                plt.xlabel("x / px",size=16)
                plt.ylabel("y / px", size=16)
                plt.tick_params(labelsize=16)

                if passed_masks != []:
                        mask_pred_pass = torch.cat(passed_masks).bool()
                        pred_segm_masks = draw_segmentation_masks(masks_background, mask_pred_pass, colors=col_seg)
                        ax3 = plt.subplot(1,3,3, sharex=ax1, sharey=ax1)
                        plt.imshow(pred_segm_masks.permute(1,2,0))
                        #plt.title("model prediction masks\n")
                        plt.xlabel("x / px", size=16)
                        plt.ylabel("y / px", size=16)
                        plt.tick_params(labelsize=16)
                else:
                        plt.subplot(1,3,3)
                        plt.title(f"no masks over threshold!")

                # to control masks easily: good -> wait 3 secs, img closes itself
                # bad -> press any key, img closes and name is noted in bad_imgs list
                print(f"image no: [{count}/{len(dataset)}]")
                count+=1
                plt.show()  #block=False to enable keypress

                #plt.pause(3)

                """key = False
                while key == False:
                        key = plt.waitforbuttonpress(3)
                        if key == True:
                                bad_imgs.append(img_name)
                plt.close()"""


print("bad_imgs are:\n")
print(bad_imgs)
bad_imgs = pd.DataFrame(bad_imgs).to_csv("bad_imgs")