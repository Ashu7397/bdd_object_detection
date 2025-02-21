import torch
import torch.nn as nn
import torch.nn.functional as F

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

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