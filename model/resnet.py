import torch.nn as nn
import torchvision.models as models


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()

        backbone = models.resnet50(pretrained=pretrained)
        in_features = backbone.fc.in_features
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_dim = in_features
        self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
       
        feat = self.encoder(x)          # (B, 512, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, 512)
        logits = self.classifier(feat)  # (B, num_classes)
        return feat, logits
