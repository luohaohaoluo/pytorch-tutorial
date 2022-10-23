import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN, MaskRCNN_ResNet50_FPN_Weights


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







