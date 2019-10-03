import torch.nn as nn
from torchvision import models

alexnet_model = models.alexnet(pretrained=True)

class AlexNetPlusLatent(nn.Module):
    def __init__(self, bits):
        super(AlexNetPlusLatent, self).__init__()
        self.bits = bits
        self.features = nn.Sequential(*list(alexnet_model.features.children()))
        self.remain = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
        self.Linear1 = nn.Linear(4096, self.bits)
        self.sigmoid = nn.Sigmoid()
        self.Linear2 = nn.Linear(self.bits, 10)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.remain(x)
        x = self.Linear1(x)
        features = self.sigmoid(x)
        result = self.Linear2(features)
        return features, result

class AutoencoderPlusLatent(nn.Module):
    def __init__(self, bits):
        super(AutoencoderPlusLatent, self).__init__()
        self.bits = bits
        self.features = nn.Sequential(*list(alexnet_model.features.children()))
        self.remain = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
        self.Linear1 = nn.Linear(4096, self.bits)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        # Decoder
        self.DLinear1 = nn.Linear(self.bits, 4096)
        self.DLinear2 = nn.Linear(4096, 9216)
        self.decoder = nn.Sequential(
            nn.Upsample(size=13),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(size=27),
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(size=55),
            nn.ConvTranspose2d(64, 3, kernel_size=11, stride=4, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.remain(x)
        x = self.Linear1(x)
        features = self.sigmoid(x)
        
        # Decoder
        d = self.DLinear1(features)
        d = self.relu(d)
        d = self.DLinear2(d)
        d = self.relu(d)
        d = d.view(d.size(0), 256, 6, 6)
        result = self.decoder(d)

        return features, result
