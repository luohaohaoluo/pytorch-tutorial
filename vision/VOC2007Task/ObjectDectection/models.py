import torch
import torchvision
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN, MaskRCNN_ResNet50_FPN_Weights

from vision.VOC2007Task.ObjectDectection.resnets import resnet50


def fasterrcnn_fpn_resnet50(num_classes):
    # features, _ = resnet50(num_classes=1000, pretrained=True)
    # backbone = features
    module = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = module.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    module.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return module


def maskrcnn_fpn_resnet50(num_classes, pretrained=True):

    module = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # features, _ = resnet50(num_classes=1000, pretrained=pretrained)
    # backbone = features
    # backbone.out_channels = 1024

    # module = MaskRCNN(backbone, num_classes=num_classes)

    in_features = module.roi_heads.box_predictor.cls_score.in_features

    module.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = module.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    module.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

    return module


if __name__ == "__main__":
    model = maskrcnn_fpn_resnet50(num_classes=20)
    print(model)
