import torch
import torch.nn as nn

class MaxFeatureMap(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        out = torch.chunk(out, 2, dim=1)
        return torch.max(out[0], out[1])


class LightCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.7):
        super().__init__()

        self.backbone = nn.Sequential(
            MaxFeatureMap(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MaxFeatureMap(nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MaxFeatureMap(nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MaxFeatureMap(nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_mfm = None
        self.dropout = nn.Dropout(p=dropout_prob)
        self.batch_norm = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, data_object, **kwargs):
        x = data_object
        if x.dim() == 3:
            x = x.unsqueeze(1)

        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        if self.fc_mfm is None:
            in_dim = features.size(1)
            self.fc_mfm = MaxFeatureMap(nn.Linear(in_dim, 256)).to(x.device)

        features = self.fc_mfm(features)
        features = self.dropout(features)
        features = self.batch_norm(features)

        logits = self.classifier(features)
        return {"logits": logits}
