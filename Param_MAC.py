import torch
import torchvision
from thop import profile, clever_format
import Network

model = Network.ARNet_3S_2N(base_plains = 64)
print(model.state_dict().keys())
input = torch.randn(1, 3, 224, 224)
qt = torch.randn(1, 64)
flops, param = profile(model, inputs = (input, qt, 3))
print("Multi-Adds = %.3fG, Params = %.3fM" % (flops / 1024.0 / 1024.0 / 1024.0, param / 1024.0 / 1024.0))
print("FLOPs are twice the number of Multi-Adds, TensorFlow calculates FLOPs instead of Multi-Adds.")