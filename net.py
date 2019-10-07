import torch.nn as nn
import torch
from torchvision import models
from torch.autograd import Variable
from itertools import chain

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

        # Freeze alexnet layers
        for child in chain(self.features.children(), self.remain.children()):
            print('Freezing layer:', child)
            for param in child.parameters():
                param.requires_grad = False
        
        # Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.bits, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(size=16, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Upsample(size=16, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Upsample(size=32, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Upsample(size=64, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Upsample(size=128, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.remain(x)
        x = self.Linear1(x)
        features = self.sigmoid(x)
        
        # Decoder
        d = self.fc_decoder(features)
        d = d.view(d.size(0), 256, 4, 4)
        result = self.decoder(d)

        return features, result

# class Decoder(nn.Module):
#     def __init__(self, bits):
#         super(Decoder, self).__init__()
#         self.fc_decoder = nn.Sequential(
#             nn.Linear(self.bits, 4096),
#             nn.ReLU(True),
#             nn.BatchNorm1d(4096),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.BatchNorm1d(4096),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.BatchNorm1d(4096)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(256),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(128),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(64),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(32),
#             nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         # Decoder
#         d = self.fc_decoder(x)
#         d = d.view(d.size(0), 256, 4, 4)
#         return self.decoder(d)

# class GANLoss(nn.Module):
#     def __init__

# class Discriminator(nn.Module):
#     def __init__(self, bits):
#         super(Discriminator, self).__init__()
#         self.bits = bits
#         self.features = nn.Sequential(*list(alexnet_model.features.children()))
#         self.remain = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
#         self.fc1 = nn.Linear(4096, 1)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.remain(x)
#         x = self.fc1(x)
#         return self.sigmoid(x)

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    def forward(self, recon_x, x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2)-logvar.exp())
        return MSE + KLD

class VAEPlusLatent(nn.Module):
    def __init__(self, bits):
        super(VAEPlusLatent, self).__init__()
        self.bits = bits
        self.features = nn.Sequential(*list(alexnet_model.features.children()))
        self.remain = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
        self.Linear1 = nn.Linear(4096, self.bits)
        self.sigmoid = nn.Sigmoid()

        # Freeze alexnet layers
        for child in chain(self.features.children(), self.remain.children()):
            for param in child.parameters():
                param.requires_grad = False

        # Variational
        # self.mu = nn.Linear(self.bits, self.bits)
        self.logvar = nn.Linear(4096, self.bits)
        
        # Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.bits, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(size=16, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Upsample(size=16, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Upsample(size=32, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Upsample(size=64, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Upsample(size=128, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        alex_last = self.remain(x)
        binary_code = self.Linear1(alex_last)
        binary_code = self.sigmoid(binary_code)

        # Variational Sample
        mu = binary_code
        logvar = self.logvar(alex_last)
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        d = self.fc_decoder(z)
        d = d.view(d.size(0), 256, 4, 4)
        result = self.decoder(d)

        return binary_code, result, mu, logvar
