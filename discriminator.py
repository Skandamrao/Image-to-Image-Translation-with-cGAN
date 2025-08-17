import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=[64, 128, 256, 512]):
        super(PatchDiscriminator, self).__init__()
        layers = []
        for idx, feature in enumerate(features):
            if idx == 0:
                layers.append(nn.Conv2d(in_channels, feature, 4, 2, 1))
            else:
                layers.append(nn.Conv2d(prev_feature, feature, 4, 2 if feature != 512 else 1, 1))
                layers.append(nn.BatchNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2))
            prev_feature = feature
        layers.append(nn.Conv2d(prev_feature, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))
