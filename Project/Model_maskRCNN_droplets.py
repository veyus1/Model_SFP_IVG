# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import numpy as np
import torch
from PIL import Image
import os
# import json
import cv2
import pandas as pd
#import xlwt

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T



class DropletsMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "raw")), key=len))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks")), key=len))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "raw", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin == xmax:
                xmin = xmin - 1
                xmax = xmax + 1
            if ymin == ymax:
                ymin = ymin - 1
                ymax = ymax + 1

            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
                  'iscrowd': iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_object_detection(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        placeholder = 1
        #transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == '__main__':
    # set random seed, for recreation:
    """torch.manual_seed(10)
    np.random.seed(10)"""

    loss_exp = []
    loss_classifier_exp = []
    loss_box_reg_exp = []
    loss_mask_exp = []
    loss_objectness_exp = []
    loss_rpn_box_reg_exp = []

    precision_recall_bbox = []
    precision_recall_segm = []

    # Enter name to save weights, losses and metrics as (e.g #train_#val_#epochs)                                    ***
    # Also add the interval for extracting data to csv (from engine.py)
    name = "1000_50_200_2_cleaned_and_flipped" #"numtrain_numeval_epochs_batchsize_other"
    num_train = 1000
    num_eval = 50
    num_epochs = 200
    batch_size = 2

    def main(precision_recall_bbox, precision_recall_segm):
        # train on the GPU or on the CPU, if a GPU is not available
        global losses_needed, accuracy_needed, metrics_needed, loss_complete, dataset_test
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # our dataset has two classes only - background, droplet, and fragments
        num_classes = 2
        # use our dataset and defined transformations
        dataset = DropletsMaskDataset('DropletsMask', get_transform(train=True))
        dataset_test = DropletsMaskDataset('DropletsMasks_eval', get_transform(train=False))

        # split the dataset in train and test set                                                                   ***
        indices_tr = torch.randperm(len(dataset)).tolist()
        indices_te = torch.randperm(len(dataset_test)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices_tr[:num_train])
        dataset_test = torch.utils.data.Subset(dataset_test, indices_te[:num_eval])

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)

        # get the model using our helper function
        model = get_model_object_detection(num_classes)

        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, weight_decay=0.0005, lr=0.005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=50,
                                                       gamma=0.05)

        # let's train it for 10 epochs                                                                              ***

        complete_dict = {}
        bbox_complete = []
        segm_complete = []

        # resume training from an earlier set of weights                                                            ***
        #model.load_state_dict(torch.load(r'Weights/weights_gpu_1000_100_10_2_transf_data.pth'))

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            # train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # Save losses
            losses_needed = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
            batch_dict = losses_needed[1]

            # compare the dict of the current epoch with the one of all past epochs and accumulate all values
            # the dict of the current epoch is created in train_one_epoch from engine.py
            for key, value in batch_dict.items():
                if key in complete_dict:
                    if isinstance(complete_dict[key], list):
                        for i in range(len(batch_dict[key])):
                            complete_dict[key].append(batch_dict[key][i])
                            #print("losses zugefügt, bestehende keys und war liste")
                    else:
                        temp_list = [complete_dict[key]]
                        temp_list.append(value)
                        complete_dict[key] = temp_list
                        #print("losses zugefügt, mit temp_list")
                else:
                    complete_dict[key] = value
                    #print("losses zugefügt, neuer key")



            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            # evaluate(model, data_loader_test, device=device)
            # Save accuracies
            coceval, precision_recall_bbox, precision_recall_segm = evaluate(model, data_loader_test, device=device)

            bbox_complete.extend(precision_recall_bbox)
            segm_complete.extend(precision_recall_segm)

        weight_path = rf'Weights\weights_{name}.pth'
        torch.save(model.state_dict(), weight_path)
        print("That's it!")

        return complete_dict, bbox_complete, segm_complete


    # Training function main()
    (loss_complete, precision_recall_bbox, precision_recall_segm) = main(precision_recall_bbox, precision_recall_segm)

    losses = pd.DataFrame.from_dict(loss_complete)
    # write data to csv
    losses_path = rf'Losses\losses_{name}.csv'
    losses.to_csv(losses_path)
    print("losses erstellt")



    prec_reca_bbox = pd.DataFrame(precision_recall_bbox, columns=['batch_no', 'prec_bbox_IoU_avg_areaall_maxdets100', 'prec_bbox_IoU_05_areaall_maxdets100',
                                                                'prec_bbox_IoU_095_areaall_maxdets100', 'prec_bbox_IoU_avg_areasmall_maxdets100',
                                                                'prec_bbox_IoU_avg_areamedium_maxdets100', 'prec_bbox_IoU_avg_arealarge_maxdets100',
                                                                'reca_bbox_IoU_avg_areaall_maxdets1', 'reca_bbox_IoU_avg_areaall_maxdets10',
                                                                'reca_bbox_IoU_avg_areaall_maxdets100', 'reca_bbox_IoU_avg_areasmall_maxdets100',
                                                                'reca_bbox_IoU_avg_areamedium_maxdets100', 'reca_bbox_IoU_avg_arealarge_maxdets100'])

    bbox_met_path = rf'Metrics\precision_recall_bbox_{name}.csv'
    prec_reca_bbox.to_csv(bbox_met_path)

    prec_reca_segm = pd.DataFrame(precision_recall_segm, columns=['batch_no', 'prec_segm_IoU_avg_areaall_maxdets100', 'prec_segm_IoU_05_areaall_maxdets100',
                                                                 'prec_segm_IoU_095_areaall_maxdets100', 'prec_segm_IoU_avg_areasmall_maxdets100',
                                                                 'prec_segm_IoU_avg_areamedium_maxdets100', 'prec_segm_IoU_avg_arealarge_maxdets100',
                                                                 'reca_segm_IoU_avg_areaall_maxdets1', 'reca_segm_IoU_avg_areaall_maxdets10',
                                                                 'reca_segm_IoU_avg_areaall_maxdets100', 'reca_segm_IoU_avg_areasmall_maxdets100',
                                                                 'reca_segm_IoU_avg_areamedium_maxdets100', 'reca_segm_IoU_avg_arealarge_maxdets100'])
    segm_met_path = rf'Metrics\precision_recall_segm_{name}.csv'
    prec_reca_segm.to_csv(segm_met_path)

