# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
if __name__ == '__main__':
    import numpy as np
    import torch
    from PIL import Image
    import os
    import json
    import cv2

    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    from engine import train_one_epoch, evaluate
    import utils
    import transforms as T



    # Read an format json file
    filejson = open('Droplets/annotation/via_project_09Mar2022_json.json')
    datajson = json.load(filejson)
    dataj = datajson
    mod_key = []
    dataj_mod = {}
    for key in datajson:
        mod_key = datajson[key]['filename']
        # Remove leading white spaces
        mod_key = mod_key.strip()  # key[:len(key)-5]
        dictt = {mod_key: datajson[key]}
        dataj_mod.update(dictt)


    class DropletsDataset(torch.utils.data.Dataset):
        def __init__(self, root, transforms):
            self.root = root
            self.transforms = transforms
            # load all image files, sorting them to
            # ensure that they are aligned
            self.imgs = list(sorted(os.listdir(os.path.join(root, "raw"))))
            # self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

        def __getitem__(self, idx):
            # load images and masks
            img_path = os.path.join(self.root, "raw", self.imgs[idx])
            # mask_path = os.path.join(self.root, "masks", self.masks[idx])
            img = Image.open(img_path).convert("RGB")
            # note that we haven't converted the mask to RGB,
            # because each color corresponds to a different instance
            # with 0 being background
            # mask = Image.open(mask_path)

            # mask = np.array(mask)
            # instances are encoded as different colors
            # obj_ids = np.unique(mask)
            # first id is the background, so remove it
            # obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set
            # of binary masks
            # masks = mask == obj_ids[:, None, None]

            filename = self.imgs[idx]
            datajs = dataj_mod[filename]['regions']
            boxes = []
            labels = []
            lbl_no = []
            for key in datajs:
                xmin = key['shape_attributes']['x']
                xmax = key['shape_attributes']['width'] + xmin
                ymin = key['shape_attributes']['y']
                ymax = key['shape_attributes']['height'] + ymin
                lbl = key['region_attributes']['type']
                if lbl == 'Regular':
                    lbl_no = 1
                if lbl == 'Deformation':
                    lbl_no = 1#2
                if lbl == 'Eruption':
                    lbl_no = 2#3
                if lbl == 'Fragmentation':
                    lbl_no = 2#4
                if lbl == 'Swell':
                    lbl_no = 1#5
                # else:
                #     lbl_no = 0

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(lbl_no)
            num_objs = len(datajs)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            # labels = torch.ones(num_objs, dtype=torch.int64)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            # masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area,
                      'iscrowd': iscrowd}

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        def __len__(self):
            return len(self.imgs)


    def get_model_object_detection(num_classes):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        # hidden_layer = 256
        # and replace the mask predictor with a new one
        # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
        #                                                    hidden_layer,
        #                                                    num_classes)

        return model


    def get_transform(train):
        transforms = [T.ToTensor()]
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)


    # Before iterating over the dataset, it’s good to see what
    # #the model expects during training and inference time on sample data.

    dataset = DropletsDataset('Droplets', get_transform(train=True))
    # dataset_test = DropletsDataset('Droplets', get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]

    # Plot example images
    boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
    img = images[0].permute(1, 2, 0).cpu().numpy()

    for box in boxes:
        cv2.rectangle(img,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0, 1))

    cv2.imshow('img', img)
    #cv2.imwrite('img.png', np.multiply(img, 255), params=None)
    cv2.waitKey()


    def main():
        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # our dataset has two classes only - background, droplet, and fragments
        num_classes = 3
        # use our dataset and defined transformations
        dataset = DropletsDataset('Droplets', get_transform(train=True))
        dataset_test = DropletsDataset('Droplets', get_transform(train=False))

        # split the dataset in train and test set
        indices = list(range(len(dataset)))
        dataset = torch.utils.data.Subset(dataset, indices[500:])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=0,
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
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # let's train it for 10 epochs
        num_epochs = 10

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)

        torch.save(model.state_dict(), 'D:/CNN/Python/Project/Weights/entire model_one class only.pth')
        print("That's it!")


    # main()
    # For Validation, plot ground truth
    #data_loader_test = torch.utils.data.DataLoader(
    #    dataset_test, batch_size=2, shuffle=True, num_workers=0,
    #    collate_fn=utils.collate_fn)
    #images, targets = next(iter(data_loader_test))
    #images = list(image for image in images)
    #targets = [{k: v for k, v in t.items()} for t in targets]
    #boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
    #img1 = images1[0].permute(1, 2, 0).cpu().numpy()

    #for box in boxes:
    #    cv2.rectangle(img1,
    #                  (box[0], box[1]),
    #                  (box[2], box[3]),
    #                  (220, 0, 0, 1))

    #cv2.imshow('img1', img1)
    #cv2.imwrite('img1.png', np.multiply(img1, 255), params=None)
    #cv2.waitKey()

    # For inference

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=3)
    model.load_state_dict(torch.load('Weights/entire model_one class only.pth'))
    model.eval()
    output = model(images)
    # Plot example images
    boxes1 = output[0]['boxes'].data.cpu().numpy().astype(np.int32)
    img1 = images[0].permute(1, 2, 0).cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()

    for box1 in boxes1:
        cv2.rectangle(img1,
                      (box1[0], box1[1]),
                      (box1[2], box1[3]),
                      (220, 0, 0, 1))

    cv2.imshow('img1', img1)
    cv2.imwrite('img_infer.png', np.multiply(img1, 255), params=None)
    cv2.waitKey()

    #
