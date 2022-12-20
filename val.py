import torch
import torchvision.transforms as transforms
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import math

import models
import dataset
from metric import calc_psnr, calc_psnrb, calc_ssim

def val_EARN(args, net, val_data_loader, save_image = False, save_image_dir = ""):
	net.eval()
	avg_comp_psnr = 0
	avg_comp_ssim = 0
	avg_comp_psnrb = 0
	avg_recover_psnr = 0
	avg_recover_ssim = 0
	avg_recover_psnrb = 0
	count = 0
	MAXI = 1.0
	if save_image:
		if not os.path.exists(save_image_dir):
			os.mkdir(save_image_dir)
	with torch.no_grad():
		with tqdm(val_data_loader) as tdl:
			for data in tdl:
				tdl.set_description("Validation")
				im_name = None
				comp_data, gt_data, qt, im_name = data
				im_name = im_name[0]
				if not net == None:
					comp_data = comp_data.cuda(device=args.gpus[0])
					qt = qt.cuda(device=args.gpus[0])
					output = net(comp_data, qt)
					if isinstance(output, tuple):
						output = output[-1]
					output = output[:,0:3,:,:]
					output = output * 0.5 + 0.5
					output[output>1] = 1
					output[output<0] = 0
					output = output.detach().cpu()
					if save_image:
						output_save = output.squeeze(0)
						output_save = transforms.ToPILImage("RGB")(output_save)
						output_save.save(os.path.join(save_image_dir, (str(count) if im_name == None else im_name) + ".png"))
					r_psnr = calc_psnr(gt_data * 0.5 + 0.5, output).item()
					r_psnrb = calc_psnrb(gt_data * 0.5 + 0.5, output).item()
					r_ssim = calc_ssim(gt_data * 0.5 + 0.5, output).item()
					avg_recover_ssim += r_ssim
					avg_recover_psnr += r_psnr
					avg_recover_psnrb += r_psnrb
				c_psnr = calc_psnr(gt_data * 0.5 + 0.5, (comp_data * 0.5 + 0.5).cpu()).item()
				c_psnrb = calc_psnrb(gt_data * 0.5 + 0.5, (comp_data * 0.5 + 0.5).cpu()).item()
				c_ssim = calc_ssim(gt_data * 0.5 + 0.5, (comp_data * 0.5 + 0.5).cpu()).item()
				avg_comp_ssim += c_ssim
				avg_comp_psnr += c_psnr
				avg_comp_psnrb += c_psnrb
				if not net == None:
					tdl.set_postfix(Recover_PSNR = r_psnr, Recover_SSIM = r_ssim, Compress_PSNR = c_psnr, Compress_SSIM = c_ssim)
				else:
					tdl.set_postfix(Compress_PSNR = c_psnr, Compress_SSIM = c_ssim)
				count += 1
	avg_comp_psnr /= count
	avg_comp_ssim /= count
	avg_comp_psnrb /= count
	avg_recover_psnr /= count
	avg_recover_ssim /= count
	avg_recover_psnrb /= count
	net.train()
	return avg_comp_psnr, avg_comp_ssim, avg_comp_psnrb, avg_recover_psnr, avg_recover_ssim, avg_recover_psnrb

if __name__ == "__main__":
	torch.set_num_threads(4)
	torch.set_num_interop_threads(4)

	# for quality in range(10, 95, 10):
	# 	net = torch.load('./G.pth').cuda()
	# 	test_set = AR_Dataset.AR_Dataset(image_dir = './LIVE1', original_format = 'bmp', phase = "Test", comp_quality = [str(quality)], channels = 3)
	# 	test_data_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = True, num_workers = 0)
	# 	avg_comp_psnr, avg_comp_ssim, avg_comp_psnrb, avg_recover_psnr, avg_recover_ssim, avg_recover_psnrb = \
	# 		test_generator(net, test_data_loader)#, save_image = True, save_image_dir = "./results/LIVE1/No_Norm")
	# 	print("JPEG Quality = " + str(quality))
	# 	print("Compressed: ")
	# 	print("PSNR = %.4f" % avg_comp_psnr)
	# 	print("PSNR-B = %.4f" % avg_comp_psnrb)
	# 	print("SSIM = %.6f" % avg_comp_ssim)
	# 	print("Recovered: ")
	# 	print("PSNR = %.4f" % avg_recover_psnr)
	# 	print("PSNR-B = %.4f" % avg_recover_psnrb)
	# 	print("SSIM = %.6f" % avg_recover_ssim)
