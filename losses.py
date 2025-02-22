import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def box_iou(boxes1, boxes2):
    """
    Calculate the Intersection over Union (IoU) of two sets of boxes.
    Boxes is a tensor of n boxes [x1, y1, x2, y2].
    Returns a tensor of shape n boxes.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def calculate_metrics(predictions_df, ground_truth_df, iou_threshold=0.5):
    """Calculates object detection metrics."""

    frame_metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    all_metrics = {'TP': 0, 'FP': 0, 'FN': 0}

    all_frames = set(predictions_df['image_id'].unique()) | set(ground_truth_df['image_id'].unique())

    for frame_name in all_frames:
        pred_frame = predictions_df[predictions_df['image_id'] == frame_name]
        gt_frame = ground_truth_df[ground_truth_df['image_id'] == frame_name]

        all_categories = set(pred_frame['category_name'].unique()) | set(gt_frame['category_name'].unique())
        for category in all_categories:
            pred_cat = pred_frame[pred_frame['category_name'] == category]
            gt_cat = gt_frame[gt_frame['category_name'] == category]

            if len(gt_cat) == 0 and len(pred_cat) == 0:
                continue

            pred_boxes = torch.tensor(np.array(pred_cat['bbox'].to_list()))
            gt_boxes = torch.tensor(np.array(gt_cat['bbox'].to_list()))

            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                iou_matrix = box_iou(pred_boxes, gt_boxes)
                matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)

                for pred_idx in range(len(pred_boxes)):
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx in range(len(gt_boxes)):
                        if iou_matrix[pred_idx, gt_idx] > best_iou and not matched_gt[gt_idx]:
                            best_iou = iou_matrix[pred_idx, gt_idx]
                            best_gt_idx = gt_idx

                    if best_iou > iou_threshold:
                        frame_metrics[frame_name]['TP'] += 1
                        all_metrics['TP'] += 1
                        matched_gt[best_gt_idx] = True
                    else:
                        frame_metrics[frame_name]['FP'] += 1
                        all_metrics['FP'] += 1
                frame_metrics[frame_name]['FN'] += torch.sum(~matched_gt).item()
                all_metrics['FN'] += torch.sum(~matched_gt).item()

            elif len(pred_boxes) > 0:
                frame_metrics[frame_name]['FP'] += len(pred_boxes)
                all_metrics['FP'] += len(pred_boxes)

            elif len(gt_boxes) > 0:
                frame_metrics[frame_name]['FN'] += len(gt_boxes)
                all_metrics['FN'] += len(gt_boxes)

    precision = all_metrics['TP'] / (all_metrics['TP'] + all_metrics['FP'] + 1e-9)
    recall = all_metrics['TP'] / (all_metrics['TP'] + all_metrics['FN'] + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'frame_metrics': dict(frame_metrics)
    }


def calculate_metrics_wrapped(pred_df, gt_df, iou_threshold, category, score_threshold):
    metrics = calculate_metrics(pred_df, gt_df, iou_threshold)
    return {
        'category': category,
        'score_threshold': score_threshold,
        'iou_threshold': iou_threshold,
        'metrics': metrics
    }

class ObjectDetectionLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2, beta=1.0, iou_threshold=0.4):
        super(ObjectDetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.iou_threshold = iou_threshold

    def focal_loss(self, inputs, targets, matched_idx, matched_gt_idx):
        num_boxes = inputs.shape[0]
        loss = torch.zeros(num_boxes, device=inputs.device)
        
        # For matched predictions
        if matched_idx.sum() > 0:
            matched_inputs = inputs[matched_idx]
            matched_targets = targets[matched_gt_idx[matched_idx]]

            # Calculate binary cross-entropy for matching predictions
            ce_loss = F.binary_cross_entropy(
                (matched_inputs == matched_targets).float(),
                torch.ones_like(matched_targets).float(),
                reduction='none'
            )

            pt = torch.exp(-ce_loss)
            loss[matched_idx] = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        # For unmatched predictions
        if (~matched_idx).sum() > 0:
            loss[~matched_idx] = torch.tensor(0.0, device=inputs.device)
        
        return loss.mean()

    def smooth_l1_loss(self, inputs, targets, matched_idx, matched_gt_idx):
        loss = torch.zeros(inputs.shape[0], 4, device=inputs.device)
        if matched_idx.sum() > 0:
            loss[matched_idx] = F.smooth_l1_loss(
                inputs[matched_idx], targets[matched_gt_idx[matched_idx]], beta=self.beta, reduction='none'
            )
        return loss.sum(dim=1).mean()

    def match_boxes(self, pred_boxes, gt_boxes):
        if len(gt_boxes) == 0:
            return torch.zeros(len(pred_boxes), dtype=torch.bool, device=pred_boxes.device), torch.zeros(len(pred_boxes), dtype=torch.long, device=pred_boxes.device)
        
        ious = box_iou(pred_boxes, gt_boxes)
        max_ious, matched_gt_idx = ious.max(dim=1)
        matched_idx = max_ious > self.iou_threshold
        return matched_idx, matched_gt_idx
        
    def forward(self, outputs, targets):
        total_cls_loss = 0
        total_bbox_loss = 0
        num_images = len(outputs)

        for i in range(num_images):
            pred_boxes = outputs[i]['boxes']
            pred_labels = outputs[i]['labels']
            gt_boxes = targets[i]['boxes']
            gt_labels = targets[i]['labels']

            matched_idx, matched_gt_idx = self.match_boxes(pred_boxes, gt_boxes)

            cls_loss = self.focal_loss(pred_labels, gt_labels, matched_idx, matched_gt_idx)
            bbox_loss = self.smooth_l1_loss(pred_boxes, gt_boxes, matched_idx, matched_gt_idx)

            total_cls_loss += cls_loss
            total_bbox_loss += bbox_loss

        avg_cls_loss = total_cls_loss / num_images
        avg_bbox_loss = total_bbox_loss / num_images
        total_loss = avg_cls_loss + avg_bbox_loss

        return total_loss, avg_cls_loss, avg_bbox_loss