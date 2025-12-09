from .vit import vit
from .vit_default import ViTBackbone

import timm

import timm
from torch.nn import Sequential
import torchvision.models as models

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    wide_resnet50_2,
    Wide_ResNet50_2_Weights,
    wide_resnet101_2,
    Wide_ResNet101_2_Weights,
)


def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == "vit":
        return vit(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == "efficientnet_v2":
        efficientnet_backbones = {
            "large": models.efficientnet_v2_l(
                weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
            ),
            "medium": models.efficientnet_v2_m(
                weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
            ),
        }
        backbone = efficientnet_backbones[cfg.MODEL.BACKBONE.SIZE]
        backbone = Sequential(*(list(backbone.children())[:-1]))
        return backbone

    # ResNet
    elif cfg.MODEL.BACKBONE.TYPE == "resnet":
        if cfg.MODEL.BACKBONE.SIZE == 50:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            backbone = Sequential(*(list(model.children())[:-2]))
            return backbone
        if cfg.MODEL.BACKBONE.SIZE == 101:
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            backbone = Sequential(*(list(model.children())[:-2]))
            return backbone
    elif cfg.MODEL.BACKBONE.TYPE == "wide_resnet":
        if cfg.MODEL.BACKBONE.SIZE == 50:
            model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
            backbone = Sequential(*(list(model.children())[:-2]))
            return backbone
        if cfg.MODEL.BACKBONE.SIZE == 101:
            model = wide_resnet101_2(weights=Wide_ResNet101_2_Weights.IMAGENET1K_V2)
            backbone = Sequential(*(list(model.children())[:-2]))
            return backbone

    # ConvNext
    elif cfg.MODEL.BACKBONE.TYPE == "convnext":
        backbone = models.convnext_large(
            weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        ).features
        return backbone
    elif cfg.MODEL.BACKBONE.TYPE == "mobilenet_v3":
        backbone = models.mobilenet_v3_large(pretrained=True).features
        return backbone
    elif cfg.MODEL.BACKBONE.TYPE == "mobilevit":
        mobilenet_backbones = {
            "small": timm.create_model("mobilevit_s", pretrained=True),
        }
        backbone = mobilenet_backbones[cfg.MODEL.BACKBONE.SIZE]
        backbone = Sequential(*list(backbone.children())[:-1])
        return backbone
    elif cfg.MODEL.BACKBONE.TYPE == "vit_default":
        backbone = ViTBackbone(
            size=cfg.MODEL.BACKBONE.SIZE,
            pretrained=True,
            freeze=False,
        )
        return backbone
    else:
        raise NotImplementedError("Backbone type is not implemented")
