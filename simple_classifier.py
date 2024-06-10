import torch.nn as nn
import torch
cuda = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = nn.Sequential(
          nn.Conv2d(1,4, kernel_size=3, padding=1),
          #nn.BatchNorm2d(4),
          nn.ReLU(),

          nn.Flatten(),
          nn.Linear(64*64*4, 6))


    def forward(self, x):
      return self.model(x.unsqueeze(1))#.squeeze([2,3])