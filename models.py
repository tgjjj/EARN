import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
import math
import time

def pixel_shuffle(scale_factor = 2):
	return torch.nn.PixelShuffle(upscale_factor = scale_factor)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, new_mean, new_std):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * new_std + new_mean

class HAdaINEncoder(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, groups = 1, w_size = 64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size = kernel_size, stride = stride, 
				padding = kernel_size // 2, groups = groups, bias = False)
		self.conv2 = nn.Conv2d(in_channels, out_channels - out_channels // 2, kernel_size = kernel_size, stride = stride, 
				padding = kernel_size // 2, groups = groups, bias = False)
		self.mean = nn.Linear(w_size, out_channels // 2)
		self.std = nn.Linear(w_size, out_channels // 2)
		self.act = torch.nn.PReLU(num_parameters = out_channels)

	def forward(self, x, qm):
		out1 = self.conv1(x)
		out2 = self.conv2(x)
		new_mean = self.mean(qm).unsqueeze(2).unsqueeze(3)
		new_std = self.std(qm).unsqueeze(2).unsqueeze(3)
		out1 = adaptive_instance_normalization(out1, new_mean, new_std)
		out = torch.cat((out1, out2), dim = 1)
		out = self.act(out)
		return out

class CAB(nn.Module): #Convolution-Activation Block
	def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, groups = 1, activation = True):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, 
				padding = kernel_size // 2, groups = groups, bias = False)
		self.act = torch.nn.PReLU(num_parameters = out_channels) if activation else None

	def forward(self, x):
		out = self.conv(x)
		if self.act is not None:
			out = self.act(out)
		return out

class DSC(nn.Module):
	def __init__(self, in_channels, out_channels, stride = 1, kernel_size = 3, channel_ratio_dwc = 4, residual = False):
		super().__init__()
		self.residual = residual
		self.cab1 = CAB(in_channels, in_channels * channel_ratio_dwc, kernel_size = 1)
		self.cab2 = CAB(in_channels * channel_ratio_dwc, in_channels * channel_ratio_dwc, kernel_size = kernel_size, 
			stride = stride, groups = in_channels * channel_ratio_dwc)
		self.cab3 = CAB(in_channels * channel_ratio_dwc, out_channels, kernel_size = 1, activation = False)

	def forward(self, x):
		out = self.cab1(x)
		out = self.cab2(out)
		out = self.cab3(out)
		if self.residual:
			out = out + x
		return out

class BS(nn.Module):
        def __init__(self, base_plains = 64):
                super().__init__()
                self.dsc1 = DSC(base_plains, base_plains * 2, stride = 2, kernel_size = 7)
                self.dsc2 = DSC(base_plains * 2, base_plains * 2, kernel_size = 7, residual = True)
                self.dsc3 = DSC(base_plains * 2, base_plains * 4, kernel_size = 7)
                self.upsample = pixel_shuffle()

        def forward(self, x):
                out = self.dsc1(x)
                out = self.dsc2(out)
                out = self.dsc3(out)
                out = self.upsample(out)
                out = out + x
                return out

class EARN(nn.Module):
	def __init__(self, in_out_channels = 3, base_channels = 64, residual = True):
		super().__init__()
		self.residual = residual
		self.encoder = HAdaINEncoder(in_out_channels, base_channels)
		self.bs1 = BS(base_channels)
		self.bs2 = BS(base_channels)
		self.bs3 = BS(base_channels)
		self.bs4 = BS(base_channels)
		self.bs5 = BS(base_channels)
		self.bs6 = BS(base_channels)
		self.decoder1 = CAB(base_channels, base_channels)
		self.decoder2 = CAB(base_channels, in_out_channels, activation = False)

	def forward(self, x, qm = None, output_positions = [2, 4, 6]):
		output_count = len(output_positions)
		results = []
		f = self.encoder(x, qm)
		for i in range(1, 7):
			bs = getattr(self, 'bs' + str(i))	
			f = bs(f)
			if i in output_positions:
				results.append(self.output(x, f))
				output_count -= 1
				if output_count == 0:
					return tuple(results)
		return tuple(results)

	def output(self, x, f):
		out = self.decoder1(f)
		out = self.decoder2(out)
		if self.residual:
			out = x + out * 0.1
		return out
