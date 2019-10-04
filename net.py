import torch.nn as nn
import torch
from torchvision import models
from torch.autograd import Variable

alexnet_model = models.alexnet(pretrained=True)

# Weight init for leaky relu
def lrelu_weight_init(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        torch.nn.init.zeros_(layer.bias)

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
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.bits, 4096),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2, True),
        )
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

    def init_weights(self):
        # Weight Init
        torch.nn.init.xavier_uniform_(self.Linear1.weight, gain=nn.init.calculate_gain('sigmoid'))
        torch.nn.init.zeros_(self.Linear1.bias)
        self.fc_decoder.apply(lrelu_weight_init)
        self.decoder.apply(lrelu_weight_init)
        torch.nn.init.xavier_uniform_(self.decoder[-2].weight, gain=nn.init.calculate_gain('tanh'))
        torch.nn.init.zeros_(self.decoder[-2].bias)
        print(self.decoder[-2])

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

        # Variational
        self.mu = nn.Linear(self.bits, self.bits)
        self.logvar = nn.Linear(self.bits, self.bits)
        
        # Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.bits, 4096),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2, True),
        )
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

    def init_weights(self):
        # Weight Init
        torch.nn.init.xavier_uniform_(self.Linear1.weight, gain=nn.init.calculate_gain('sigmoid'))
        torch.nn.init.zeros_(self.Linear1.bias)
        torch.nn.init.xavier_uniform_(self.mu.weight, gain=1)
        torch.nn.init.zeros_(self.mu.bias)
        torch.nn.init.xavier_uniform_(self.logvar.weight, gain=1)
        torch.nn.init.zeros_(self.logvar.bias)
        self.fc_decoder.apply(lrelu_weight_init)
        self.decoder.apply(lrelu_weight_init)
        torch.nn.init.xavier_uniform_(self.decoder[-2].weight, gain=nn.init.calculate_gain('tanh'))
        torch.nn.init.zeros_(self.decoder[-2].bias)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.remain(x)
        x = self.Linear1(x)
        features = self.sigmoid(x)

        # Variational Sample
        mu = self.mu(features)
        logvar = self.logvar(features)
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        d = self.fc_decoder(z)
        d = d.view(d.size(0), 256, 4, 4)
        result = self.decoder(d)

        return features, result, mu, logvar
