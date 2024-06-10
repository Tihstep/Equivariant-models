import torch.nn as nn
import torch
cuda = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.encode_backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.linear_mean =  nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=200, bias=True)
        )

        self.linear_log_var =  nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=200, bias=True)
        )

        self.training = True

    def forward(self, x):
        x = x.unsqueeze(1)
        h_ = self.encode_backbone(x)
        h_ = torch.flatten(h_, 1)
        mean = self.linear_mean(h_)
        log_var = self.linear_log_var(h_)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)

        self.decode_backbone = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1, 32, kernel_size=3, stride = 1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, out_channels=1,
                      kernel_size= 3, padding= 1),
            nn.Flatten(),
            nn.Linear(64*64, 9)
        )

    def forward(self, x):
        h = self.FC_hidden(x)
        x_hat = self.decode_backbone(h.view(256, 1, 64, 64))
        x_hat = torch.tanh(x_hat)
        return x_hat


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)
        z = mean + var*epsilon
        return z


    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat            = self.Decoder(z)
        return x_hat.squeeze(1), mean, log_var, z