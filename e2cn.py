import torch

from e2cnn import gspaces
from e2cnn import nn
cuda = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

r2_act = gspaces.Flip2dOnR2()

feat_type_in = nn.FieldType(r2_act, 1*[r2_act.trivial_repr])
feat_type_hid = nn.FieldType(r2_act, 4*[r2_act.trivial_repr])
feat_type_out = nn.FieldType(r2_act, [r2_act.trivial_repr])



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = nn.SequentialModule(
          nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=3, padding=1),
          #nn.InnerBatchNorm(feat_type_hid),
          nn.ReLU(feat_type_hid),
)
        self.MLP = torch.nn.Sequential(
          torch.nn.Flatten(),
          torch.nn.Linear(64*64*4, 6)
          )
#nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
    def forward(self, x):
      x = x.unsqueeze(1)
      x = nn.GeometricTensor(x, feat_type_in)
      x = self.model(x)
      x = x.tensor
      return self.MLP(x)