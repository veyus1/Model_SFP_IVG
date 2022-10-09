import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import utils
from Model_maskRCNN_droplets import get_transform, get_model_object_detection
from Model_maskRCNN_different_backbone import get_model_w_new_bb
import os
from PIL import Image
import onnxruntime
import time

# NUR WÄHREND DEM TESTEN: random seed setzen
"""torch.manual_seed(0)
np.random.seed(0)"""

"""Takes all images from Inference folder, creates predictions for raw images and saves the predictions in Predictions folder"""


class DropletsMaskDataset_Inf(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "raw"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "raw", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        image_id = torch.tensor([idx])

        target = {"image_id": image_id}

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    batch_size = 1
    dataset = DropletsMaskDataset_Inf('Inference', get_transform(train=False))  # set root folder
    indices = list(range(len(dataset)))
    #indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:])
    #dataset = dataset_test
    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)




    #model = get_model_w_new_bb(num_classes=2)
    #model.load_state_dict(torch.load('Weights/weights_300_50_5_res18_2b_testgrayscale.pth'))
    # change weights to run desired onnx model:
    model = onnxruntime.InferenceSession("Weights/ONNX/weights_500_50_5_res18_2b_final.onnx")


    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    #model.to("cpu"), this is covered in to:numpy function
    # dont need model.eval() with onnx model

    count = 1
    max_imgs = len(dataset)
    start_time = time.time()
    for image, targets in data_loader:

            #outputs = model(image) # forward pass durch Modell, zeitaufwändig

            # the onnx model expects a specific size: (batch_size,1,1024,144)
            image = image[0].unsqueeze(0)

            # compute onnx prediction
            ort_inputs = {model.get_inputs()[0].name: to_numpy(image)}
            outputs = model.run(None, ort_inputs)

            for i in range(batch_size):

                # the onnx model expects a specific size: (batch_size,1,1024,144)
                #image = image[i].unsqueeze(0)

                # get masks and their scores, if they are good enough append to passed_masks list
                output = outputs
                id = targets[i]["image_id"].item()
                masks = torch.from_numpy(output[3])
                scores = torch.from_numpy(output[2])
                boxes = torch.from_numpy(output[0])
                threshold = 0.95

                passed_masks = []
                mask_pred_pass = torch.tensor((len(passed_masks), masks.size(dim=2), masks.size(dim=3)))

                # use non maximum supression (nms) to find boxes/masks with best iou:
                nms_boxes = []
                nms_masks = []
                nms_indices = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=1)
                for j in nms_indices:
                        nms_boxes.append(boxes[j])
                        nms_masks.append(masks[j])

                # for semantic segmentation (rn just for experimenting)
                # create one mask that contains the most probable values
                #total_mask = torch.zeros(1,masks.size(dim=2), masks.size(dim=3))
                #total_mask = torch.cat(tuple(nms_masks)).unsqueeze(dim=1).max(dim=0)[0].unsqueeze(dim=0)


                # concatenate all boxes and masks that pass nms threshold criteria (N)
                if len(nms_boxes) > 0:
                        boxes = torch.cat(nms_boxes).view(nms_indices.size(dim=0), 4)   # Nx4
                        masks = torch.cat(nms_masks).view(nms_indices.size(dim=0), 1, masks.size(dim=2),
                                                                                       masks.size(dim=3))  # Nx1xHxW

                        # use this as mask for sem segm (otherwise statement above):
                        #masks = total_mask

                # if mask value is below threshold set to zero and append to passed_masks:
                for mask in masks:
                        mask[mask < threshold] = 0
                        passed_masks.append(mask)

                # use torchvision io functions to draw masks on background image
                masks_background = torch.zeros((3,torch.from_numpy(output[3]).size(dim=2),
                                                torch.from_numpy(output[3]).size(dim=3)), dtype=torch.uint8)

                col_seg = []
                for c in range(len(passed_masks)):
                        col_seg.append((255-c, 255-c,255-c))

                img_name = dataset.dataset.imgs[id]
                img_name = img_name.split(sep=".")[0]
                format = "tif"

                if passed_masks != []:
                        mask_pred_pass = torch.cat(passed_masks).bool()
                        pred_segm_masks = draw_segmentation_masks(masks_background, mask_pred_pass, colors=col_seg)
                        #ax3 = plt.subplot(1,3,3, sharex=ax1, sharey=ax1)
                        #a = plt.imshow(pred_segm_masks.permute(1,2,0))

                        save_path = os.path.join("Inference/Predictions", str("pred_"+ img_name+"." + format))
                        save_path_ov = os.path.join("Inference/raw_vis", str(img_name + "." + format))
                        im_form = np.array(pred_segm_masks.permute(1,2,0))
                        #plt.imsave(save_path,im_form,format= format )

                        # convert image to preferred format (P, L, RGB...):
                        img = Image.fromarray(im_form).convert("L").save(save_path)
                        img_ov = np.array(torch.mul(image[i].permute(1,2,0),255).to(torch.uint8))
                        img_ov = Image.fromarray(img_ov, mode="RGB").save(save_path_ov)
                        print(f"[{count}/{max_imgs}] pred for {img_name} saved")


                else:
                        print(f"[{count}/{max_imgs}] no segmentation masks found for {img_name}")

                count+=1

                # comment out to see every prediction before saving it
                #plt.title(f"model prediction masks for {img_name}")
                #plt.show()
    end_time = time.time()
    print(f"total inference time for {max_imgs} images: {(end_time-start_time):.4f} seconds")

    """model.eval()
    x = torch.rand(batch_size, 1,1024,144)
    image, _ = dataset[0]
    image = image.unsqueeze(0)
    #out = model(x)"""

    # export
    """torch.onnx.export(model,x,"grey_model_test.onnx", export_params=True,
                      opset_version=11, do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input":{0:"batch_size"},
                                    "output":{0:"batch_size"}})"""



    """onnx_model = onnx.load()
    onnx.checker.check_model(onnx_model)

            sess = onnxruntime.InferenceSession("grey_model_test.onnx")

            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

            # compute onnx prediction
            ort_inputs = {sess.get_inputs()[0].name: to_numpy(image)}
            ort_outs = sess.run(None, ort_inputs)
            print("ok")
            # compare ONNX Runtime and PyTorch results"""
