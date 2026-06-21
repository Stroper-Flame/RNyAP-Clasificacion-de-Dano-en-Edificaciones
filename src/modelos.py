import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from src.configuracion import (
    NUM_CLASSES,
    FEATURE_DIM,
    PRETRAINED
)


class SiameseResNet18(nn.Module):

    def __init__(self):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if PRETRAINED
            else None
        )

        backbone = resnet18(
            weights=weights
        )

        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.classifier = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, pre_img, post_img):

        imagenes = torch.cat(
            (pre_img, post_img),
            dim=0
        )

        features = self.feature_extractor(
            imagenes
        )

        features = torch.flatten(
            features,
            start_dim=1
        )

        f_pre, f_post = torch.chunk(
            features,
            chunks=2,
            dim=0
        )

        diff = torch.abs(
            f_post - f_pre
        )

        output = self.classifier(
            diff
        )

        return output