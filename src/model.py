from __future__ import annotations
from typing import Dict, Optional

import torch
from torch import nn
from torchvision import models


class MultiHeadFashionClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_class_labels: int,
        num_gender_labels: int,
        num_color_labels: int,
        dropout: float = 0.25,
        pretrained: bool = True,
        num_usage_labels: Optional[int] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        elif backbone_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_large(weights=weights)
            in_features = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError("Unsupported backbone. Choose: resnet50, efficientnet_b0, mobilenet_v3_large")
        self.backbone = backbone
        self.shared_dropout = nn.Dropout(dropout)
        self.class_head = nn.Linear(in_features, num_class_labels)
        self.gender_head = nn.Linear(in_features, num_gender_labels)
        self.color_head = nn.Linear(in_features, num_color_labels)
        self.usage_head = nn.Linear(in_features, num_usage_labels) if num_usage_labels else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.shared_dropout(features)
        outputs = {
            "class": self.class_head(features),
            "gender": self.gender_head(features),
            "color": self.color_head(features),
        }
        if self.usage_head is not None:
            outputs["usage"] = self.usage_head(features)
        return outputs


class ONNXExportWrapper(nn.Module):
    def __init__(self, model: MultiHeadFashionClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        if self.model.usage_head is not None:
            return outputs["class"], outputs["gender"], outputs["color"], outputs["usage"]
        return outputs["class"], outputs["gender"], outputs["color"]


def build_model_from_label_info(label_info: Dict, backbone_name: str, dropout: float = 0.25, pretrained: bool = True) -> MultiHeadFashionClassifier:
    usage_info = label_info.get("usage")
    num_usage = usage_info["num_classes"] if usage_info else None
    return MultiHeadFashionClassifier(
        backbone_name=backbone_name,
        num_class_labels=label_info["class"]["num_classes"],
        num_gender_labels=label_info["gender"]["num_classes"],
        num_color_labels=label_info["color"]["num_classes"],
        dropout=dropout,
        pretrained=pretrained,
        num_usage_labels=num_usage,
    )
