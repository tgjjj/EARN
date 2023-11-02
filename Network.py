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

def adaptive_affine_transformation(content_feat, new_mean, new_std):
    return content_feat * new_std + new_mean

class CNL(nn.Module):
	def __init__(self, in_plains, out_plains, kernel_size = 3, stride = 1, groups = 1, dilation = 1, norm_type = 'AdaIN', w_size = 64, non_linear = True):
		super().__init__()
		assert norm_type in ['AdaIN', 'IN', 'HIN', 'AT', ''], 'invalid norm_type for CNL'
		self.norm_type = norm_type
		self.conv = nn.Conv2d(in_plains, out_plains, kernel_size = kernel_size, stride = stride, 
		 		padding = (kernel_size // 2) * dilation if isinstance(kernel_size, int) else ((kernel_size[0] // 2) * dilation, (kernel_size[1] // 2) * dilation), 
		 		groups = groups, bias = False, dilation = dilation)
		if norm_type == 'AdaIN' or norm_type == 'AT':
			self.at_mean = nn.Linear(w_size, out_plains)
			self.at_std = nn.Linear(w_size, out_plains)
		elif norm_type == 'IN':
			self.norm = IN(out_plains)
		elif norm_type == 'HIN':
			self.norm = IN(out_plains // 2, affine = True)
		self.nl = torch.nn.PReLU(num_parameters = out_plains) if non_linear else None

	def forward(self, x, w = None):
		out = self.conv(x)
		if self.norm_type == 'AdaIN':
			mean = self.at_mean(w).unsqueeze(2).unsqueeze(3)
			std = self.at_std(w).unsqueeze(2).unsqueeze(3)
			out = adaptive_instance_normalization(out, mean, std)
		elif self.norm_type == 'IN':
			out = self.norm(out)
		elif self.norm_type == 'HIN':
			c = out.size(1)
			out1, out2 = out.split([c // 2, c - c // 2], dim = 1)
			out1 = self.norm(out1)
			out = torch.cat((out1, out2), dim = 1)
		elif self.norm_type == 'AT':
			mean = self.at_mean(w).unsqueeze(2).unsqueeze(3)
			std = self.at_std(w).unsqueeze(2).unsqueeze(3)
			out = adaptive_affine_transformation(out, mean, std)
		if self.nl is not None:
			out = self.nl(out)
		return out

class Conv_Block(nn.Module):
	def __init__(self, in_plains, out_plains, stride = 1, kernel_size = 3, DWC = True, channel_ratio_dwc = 4, norm_type = 'AdaIN', w_size = 64):
		super().__init__()
		self.DWC = DWC
		if self.DWC:
			self.cnl1 = CNL(in_plains, in_plains * channel_ratio_dwc, kernel_size = 1, norm_type = norm_type, w_size = w_size)
			self.cnl2 = CNL(in_plains * channel_ratio_dwc, in_plains * channel_ratio_dwc, kernel_size = kernel_size, 
				stride = stride, groups = in_plains * channel_ratio_dwc, norm_type = norm_type, w_size = w_size)
			self.cnl3 = CNL(in_plains * channel_ratio_dwc, out_plains, kernel_size = 1, norm_type = norm_type, w_size = w_size, non_linear = False)
		else:
			self.cnl = CNL(in_plains, out_plains, stride = stride, kernel_size = kernel_size, norm_type = norm_type, w_size = w_size)

	def forward(self, x, w = None):
		if self.DWC:
			out = self.cnl1(x, w)
			out = self.cnl2(out, w)
			out = self.cnl3(out, w)
		else:
			out = self.cnl(x, w)
		if out.size() == x.size():
			out = out + x
		return out

class Basic_Net(nn.Module):
        def __init__(self, base_plains = 64, DWC = True, norm_type = 'AdaIN'):
                super().__init__()
                self.block1 = Conv_Block(base_plains, base_plains * 2, stride = 2, kernel_size = 7 if DWC else 3, DWC = DWC, norm_type = norm_type)
                self.block2 = Conv_Block(base_plains * 2, base_plains * 2, kernel_size = 7 if DWC else 3, DWC = DWC, norm_type = norm_type)
                self.block3 = Conv_Block(base_plains * 2, base_plains * 4, kernel_size = 7 if DWC else 3, DWC = DWC, norm_type = norm_type)
                self.upsample = pixel_shuffle()

        def forward(self, x, w = None):
                short_cut0 = x
                out = self.block1(x, w)
                out = self.block2(out, w)
                out = self.block3(out, w)
                out = self.upsample(out)
                out = out + short_cut0
                return out

class ARNet_3S_2N(nn.Module):
	# norm_mode = Full / Head / Half_Head / No
	# DWC = True / False
	# norm_type = AdaIN / IN
	def __init__(self, in_out_plains = 3, base_plains = 64, DWC = True, norm_mode = 'Half_Head', norm_type = 'AdaIN', residual = True):
		super().__init__()
		self.residual = residual
		self.head1 = CNL(in_out_plains, base_plains // 2, norm_type = norm_type if norm_mode == 'Head' or norm_mode == 'Full' else '')
		self.head2 = CNL(in_out_plains, base_plains // 2, norm_type = norm_type if norm_mode != 'No' else '')
		self.net1 = Basic_Net(base_plains, DWC, '' if norm_mode != 'Full' else norm_type)
		self.net2 = Basic_Net(base_plains, DWC, '' if norm_mode != 'Full' else norm_type)
		self.net3 = Basic_Net(base_plains, DWC, '' if norm_mode != 'Full' else norm_type)
		self.net4 = Basic_Net(base_plains, DWC, '' if norm_mode != 'Full' else norm_type)
		self.net5 = Basic_Net(base_plains, DWC, '' if norm_mode != 'Full' else norm_type)
		self.net6 = Basic_Net(base_plains, DWC, '' if norm_mode != 'Full' else norm_type)
		self.tail0 = CNL(base_plains, base_plains, norm_type = '' if norm_mode != 'Full' else norm_type)
		self.tail1 = CNL(base_plains, in_out_plains, norm_type = '', non_linear = False)

	def forward(self, x, qt = None, stage_num = 3, test_mode = False):
		time_start = time.time()
		rs = []
		f1 = self.head1(x, qt)
		f2 = self.head2(x, qt)
		f = torch.cat((f1, f2), dim = 1)
		if stage_num >= 1:
			f = self.net1(f, qt)
			f = self.net2(f, qt)
			if not test_mode or stage_num == 1:
				r1 = self.tail1(self.tail0(f, qt))
				if self.residual:
					r1 = x + r1 * 0.1
				rs.append(r1)
		if stage_num >= 2:
			f = self.net3(f, qt)
			f = self.net4(f, qt)
			if not test_mode or stage_num == 2:
				r2 = self.tail1(self.tail0(f, qt))
				if self.residual:
					r2 = x + r2 * 0.1
				rs.append(r2)
		if stage_num >= 3:
			f = self.net5(f, qt)
			f = self.net6(f, qt)
			if not test_mode or stage_num == 3:
				r3 = self.tail1(self.tail0(f, qt))
				if self.residual:
					r3 = x + r3 * 0.1
				rs.append(r3)
		return tuple(rs)