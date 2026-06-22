import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from src.configuracion import NUM_CLASSES, FEATURE_DIM, PRETRAINED

#Red siamesa: extrae features de pre/post disaster, calcula diferencia y clasifica
class SiameseResNet18(nn.Module):

    #Inicializa el backbone ResNet18 y el clasificador
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if PRETRAINED else None
        backbone = resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

    #Concatena pre y post, extrae features, calcula |diff| y clasifica
    def forward(self, pre_img, post_img):
        x = torch.cat((pre_img, post_img), dim=0)
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        f_pre, f_post = torch.chunk(x, chunks=2, dim=0)
        return self.classifier(torch.abs(f_post - f_pre))
