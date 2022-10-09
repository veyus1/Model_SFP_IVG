import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import numpy as np
import copy


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )


    loss_exp = []
    loss_classifier_exp = []
    loss_box_reg_exp = []
    loss_mask_exp = []
    loss_objectness_exp = []
    loss_rpn_box_reg_exp = []
    batches_logged = []
    epochs_logged = []
    curr_batch_no = 0

    for images, targets in metric_logger.log_every(data_loader, print_freq, header): # nimmt imgs und targets einer minibatch)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):

            loss_dict = model(images, targets) # forward pass
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward() # backward pass
            optimizer.step() # and one step of optimizer

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # taking all wanted metrics out of metric_logger every 10th batches                                         ***
        if curr_batch_no % 1 == 0:
            loss_exp.append(metric_logger.meters['loss'].avg)
            loss_classifier_exp.append(metric_logger.meters['loss_classifier'].avg)
            loss_box_reg_exp.append(metric_logger.meters['loss_box_reg'].avg)
            loss_mask_exp.append(metric_logger.meters['loss_mask'].avg)
            loss_objectness_exp.append(metric_logger.meters['loss_objectness'].avg)
            loss_rpn_box_reg_exp.append(metric_logger.meters['loss_rpn_box_reg'].avg)
            batches_logged.append(curr_batch_no)
            epochs_logged.append(epoch)
        curr_batch_no += 1

        # and updating the losses_list with them
        losses_dict = {'epoch': epochs_logged, 'batch_no': batches_logged, 'loss_exp': loss_exp, 'loss_classifier_exp': loss_classifier_exp, 'loss_box_reg_exp': loss_box_reg_exp,
                          'loss_mask_exp': loss_mask_exp, 'loss_objectness_exp': loss_objectness_exp, 'loss_rpn_box_reg_exp': loss_rpn_box_reg_exp}
        # a list with all the loss value-entries of all batches in one epoch



    return metric_logger, losses_dict


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    precision_recall_bbox = []
    precision_recall_segm = []
    i = 0

    for images, targets in metric_logger.log_every(data_loader, 1, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        coco_evaluator2 = copy.deepcopy(coco_evaluator)
        coco_evaluator2.synchronize_between_processes()
        coco_evaluator2.accumulate()

        # change number to append data every x-th iteration (dont change i)                                    ***
        if i % 1 == 0:
            precision_recall_bbox.append(
                [i, np.mean(np.mean(coco_evaluator2.coco_eval['bbox'].eval['precision'], axis=0), axis=0)[0][0][2],
                 # prec_bbox_IoU_avg_areaall_maxdets100
                 np.mean(coco_evaluator2.coco_eval['bbox'].eval['precision'][0], axis=0)[0][0][2],
                 # prec_bbox_IoU_05_areaall_maxdets100
                 np.mean(coco_evaluator2.coco_eval['bbox'].eval['precision'][5], axis=0)[0][0][2],
                 # prec_bbox_IoU_095_areaall_maxdets100 --hier müsste 075 (?)
                 np.mean(np.mean(coco_evaluator2.coco_eval['bbox'].eval['precision'], axis=0), axis=0)[0][1][2],
                 # prec_bbox_IoU_avg_areasmall_maxdets100
                 np.mean(np.mean(coco_evaluator2.coco_eval['bbox'].eval['precision'], axis=0), axis=0)[0][2][2],
                 # prec_bbox_IoU_avg_areamedium_maxdets100
                 np.mean(np.mean(coco_evaluator2.coco_eval['bbox'].eval['precision'], axis=0), axis=0)[0][3][2],
                 # prec_bbox_IoU_avg_arealarge_maxdets100
                 np.mean(coco_evaluator2.coco_eval['bbox'].eval['recall'], axis=0)[0][0][0],
                 # reca_bbox_IoU_avg_areaall_maxdets1
                 np.mean(coco_evaluator2.coco_eval['bbox'].eval['recall'], axis=0)[0][0][1],
                 # reca_bbox_IoU_avg_areaall_maxdets10
                 np.mean(coco_evaluator2.coco_eval['bbox'].eval['recall'], axis=0)[0][0][2],
                 # reca_bbox_IoU_avg_areaall_maxdets100
                 np.mean(coco_evaluator2.coco_eval['bbox'].eval['recall'], axis=0)[0][1][2],
                 # reca_bbox_IoU_avg_areasmall_maxdets100
                 np.mean(coco_evaluator2.coco_eval['bbox'].eval['recall'], axis=0)[0][2][2],
                 # reca_bbox_IoU_avg_areamedium_maxdets100
                 np.mean(coco_evaluator2.coco_eval['bbox'].eval['recall'], axis=0)[0][3][2]])
                 # reca_bbox_IoU_avg_arealarge_maxdets100

            precision_recall_segm.append(
                [i, np.mean(np.mean(coco_evaluator2.coco_eval['segm'].eval['precision'], axis=0), axis=0)[0][0][2],
                 # prec_segm_IoU_avg_areaall_maxdets100
                 np.mean(coco_evaluator2.coco_eval['segm'].eval['precision'][0], axis=0)[0][0][2],
                 # prec_segm_IoU_05_areaall_maxdets100
                 np.mean(coco_evaluator2.coco_eval['segm'].eval['precision'][5], axis=0)[0][0][2],
                 # prec_segm_IoU_095_areaall_maxdets100
                 np.mean(np.mean(coco_evaluator2.coco_eval['segm'].eval['precision'], axis=0), axis=0)[0][1][2],
                 # prec_segm_IoU_avg_areasmall_maxdets100
                 np.mean(np.mean(coco_evaluator2.coco_eval['segm'].eval['precision'], axis=0), axis=0)[0][2][2],
                 # prec_segm_IoU_avg_areamedium_maxdets100
                 np.mean(np.mean(coco_evaluator2.coco_eval['segm'].eval['precision'], axis=0), axis=0)[0][3][2],
                 # prec_segm_IoU_avg_arealarge_maxdets100
                 np.mean(coco_evaluator2.coco_eval['segm'].eval['recall'], axis=0)[0][0][0],
                 # reca_segm_IoU_avg_areaall_maxdets1
                 np.mean(coco_evaluator2.coco_eval['segm'].eval['recall'], axis=0)[0][0][1],
                 # reca_segm_IoU_avg_areaall_maxdets10
                 np.mean(coco_evaluator2.coco_eval['segm'].eval['recall'], axis=0)[0][0][2],
                 # reca_segm_IoU_avg_areaall_maxdets100
                 np.mean(coco_evaluator2.coco_eval['segm'].eval['recall'], axis=0)[0][1][2],
                 # reca_segm_IoU_avg_areasmall_maxdets100
                 np.mean(coco_evaluator2.coco_eval['segm'].eval['recall'], axis=0)[0][2][2],
                 # reca_segm_IoU_avg_areamedium_maxdets100
                 np.mean(coco_evaluator2.coco_eval['segm'].eval['recall'], axis=0)[0][3][2]])
                 # reca_segm_IoU_avg_arealarge_maxdets100

            #coco_evaluator2.summarize()
        i += 1



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    #print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    print("-----Metrics nach einer Epoche:-----")
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator, precision_recall_bbox, precision_recall_segm

