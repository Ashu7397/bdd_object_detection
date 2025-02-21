import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from losses import ObjectDetectionLoss


class ObjectDetectionModel(pl.LightningModule):
    """
    A PyTorch Lightning module for object detection models.

    Args:
        model (nn.Module): The object detection model.
        learning_rate (float): The learning rate for the optimizer.
        num_classes (int): Number of classes in the dataset.
    """

    def __init__(self, model: nn.Module, learning_rate: float, num_classes: int):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.save_hyperparameters()

        self.loss_function = ObjectDetectionLoss(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
    def main_forward_step(self, batch, batch_idx):
        images, targets = batch
        outputs_list = self.model(images, targets)        
        return outputs_list
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.main_forward_step(batch, batch_idx)
        loss = sum(loss for loss in loss_dict.values())
        self.log(f'train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs_list = self.main_forward_step(batch, batch_idx)
        loss, cls_loss, reg_loss = self.loss_function(outputs_list, targets)
        self.log(f'val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    


def get_pretrained_model(num_classes):
    """
    Returns a pretrained Faster R-CNN model with ResNet-50 backbone
    suitable for BDD100K object detection.
    num_classes (int): Number of classes in the dataset.
    """

    resnet = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT")
    backbone = nn.Sequential(*list(resnet.children())[:-2]) # Remove last 2 layers (avgpool and fc)

    backbone.out_channels = 2048  # ResNet50 output channels before the last layer is 2048

    # Set backbone to non-trainable
    for param in backbone.parameters():
        param.requires_grad = False

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model