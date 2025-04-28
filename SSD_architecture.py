import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights

class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes

        # Load the pre-trained VGG16 model
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.features = nn.ModuleList(vgg[:30])  # Use up to the conv5_3 layer

        # Additional layers for SSD
        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True)
            )
        ])

        # Localization and class prediction layers
        self.loc = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),  # 4 default boxes
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),  # 6 default boxes
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),  # 6 default boxes
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),  # 6 default boxes
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),  # 4 default boxes
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)   # 4 default boxes
        ])

        self.conf = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        locs = []
        confs = []

        # Apply base network
        for k in range(len(self.features)):
            x = self.features[k](x)
        
        # Apply localization and confidence layers on conv4_3 and conv7
        locs.append(self.loc[0](x).permute(0, 2, 3, 1).contiguous())
        confs.append(self.conf[0](x).permute(0, 2, 3, 1).contiguous())

        for (i, layer) in enumerate(self.extras):
            x = layer(x)
            locs.append(self.loc[i+1](x).permute(0, 2, 3, 1).contiguous())
            confs.append(self.conf[i+1](x).permute(0, 2, 3, 1).contiguous())

        # Reshape and concatenate predictions
        locs = torch.cat([o.view(o.size(0), -1) for o in locs], 1)
        confs = torch.cat([o.view(o.size(0), -1) for o in confs], 1)

        locs = locs.view(locs.size(0), -1, 4)
        confs = confs.view(confs.size(0), -1, self.num_classes)

        return locs, confs


num_classes = 21
ssd = SSD(num_classes)
x = torch.randn(1, 3, 300, 300)
locs, confs = ssd(x)
print("Localization predictions:", locs.size())
print("Confidence predictions:", confs.size())
