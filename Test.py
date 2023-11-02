import torch
import torchvision.transforms as transforms
import torchvision
import os
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import math

import Network
import AR_Dataset

def calc_psnr(target, input):
    mse = torch.nn.functional.mse_loss(input, target, reduction='none')
    mse = mse.view(mse.shape[0], -1).mean(1)
    return 10 * torch.log10(1 / mse)

def blocking_effect_factor(im):
    block_size = 8

    block_horizontal_positions = torch.arange(7,im.shape[3]-1,8)
    block_vertical_positions = torch.arange(7,im.shape[2]-1,8)

    horizontal_block_difference = ((im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1])**2).sum(3).sum(2).sum(1)
    vertical_block_difference = ((im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :])**2).sum(3).sum(2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(torch.arange(0,im.shape[3]-1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(torch.arange(0,im.shape[2]-1), block_vertical_positions)

    horizontal_nonblock_difference = ((im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1])**2).sum(3).sum(2).sum(1)
    vertical_nonblock_difference = ((im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :])**2).sum(3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3]//block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2]//block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (n_boundary_horiz + n_boundary_vert)


    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef


def calc_psnrb(target, input):
    total = 0
    for c in range(input.shape[1]):
        mse = torch.nn.functional.mse_loss(input[:, c:c+1, :, :], target[:, c:c+1, :, :], reduction='none')
        bef = blocking_effect_factor(input[:, c:c+1, :, :])

        mse = mse.view(mse.shape[0], -1).mean(1)
        total += 10 * torch.log10(1 / (mse + bef))

    return total / input.shape[1]

def ssim_single(target, input, device=None):
    C1 = 0.01**2
    C2 = 0.03**2

    filter = torch.ones(1, 1, 8, 8) / 64

    if device is not None:
        filter = filter.to(device)

    mu_i = torch.nn.functional.conv2d(input, filter)
    mu_t = torch.nn.functional.conv2d(target, filter)

    var_i = torch.nn.functional.conv2d(input**2, filter) - mu_i**2
    var_t = torch.nn.functional.conv2d(target**2, filter) - mu_t**2
    cov_it = torch.nn.functional.conv2d(target*input, filter) - mu_i * mu_t

    ssim_blocks = ((2 * mu_i * mu_t + C1) * (2 * cov_it + C2)) / ((mu_i**2 + mu_t**2 + C1) * (var_i + var_t + C2))
    return ssim_blocks.view(input.shape[0], -1).mean(1)


def calc_ssim(target, input, device=None):
    total = 0
    for c in range(target.shape[1]):
        total += ssim_single(target[:, c:c+1, :, :], input[:, c:c+1, :, :], device)

    return total / target.shape[1]

def test_model(net, test_data_loader, save_image = False, save_image_dir = ""):
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
		# for data in test_data_loader:
		with tqdm(test_data_loader) as tdl:
			for data in tdl:
				tdl.set_description("Test")
				im_name = None
				comp_data, gt_data, qt, im_name = data
				im_name = im_name[0]
				if not net == None:
					comp_data = comp_data.cuda(device = 0)
					qt = qt.cuda(device = 0)
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
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

	torch.set_num_threads(4)
	torch.set_num_interop_threads(4)
	tf.config.threading.set_inter_op_parallelism_threads(4)
	tf.config.threading.set_intra_op_parallelism_threads(4)

	for quality in range(10, 95, 10):
		net = torch.load('./G.pth').cuda(device = 0)
		test_set = AR_Dataset.AR_Dataset(image_dir = './LIVE1', original_format = 'bmp', phase = "Test", comp_quality = [str(quality)], channels = 3)
		test_data_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = True, num_workers = 0)
		avg_comp_psnr, avg_comp_ssim, avg_comp_psnrb, avg_recover_psnr, avg_recover_ssim, avg_recover_psnrb = \
			test_model(net, test_data_loader)#, save_image = True, save_image_dir = "./results/LIVE1/No_Norm")
		print("JPEG Quality = " + str(quality))
		print("Compressed: ")
		print("PSNR = %.4f" % avg_comp_psnr)
		print("PSNR-B = %.4f" % avg_comp_psnrb)
		print("SSIM = %.6f" % avg_comp_ssim)
		print("Recovered: ")
		print("PSNR = %.4f" % avg_recover_psnr)
		print("PSNR-B = %.4f" % avg_recover_psnrb)
		print("SSIM = %.6f" % avg_recover_ssim)
