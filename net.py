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
        
        # Decoder
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.fc1 = nn.Linear(self.bits, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.remain(x)
        x = self.Linear1(x)
        features = self.sigmoid(x)
        
        # Decoder
        d = self.fc1(features)
        d = self.lrelu(d)
        d = self.fc2(d)
        d = self.lrelu(d)
        d = self.fc3(d)
        d = self.lrelu(d)
        d = d.view(d.size(0), 256, 4, 4)
        result = self.decoder(d)

        return features, result
