from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import math

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

if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

	torch.set_num_threads(4)
	torch.set_num_interop_threads(4)
	tf.config.threading.set_inter_op_parallelism_threads(4)
	tf.config.threading.set_intra_op_parallelism_threads(4)

	imgs_dir_GT = './LIVE1/bmp/'
	imgs_dir_Test = ['LIVE1/jpg10']#, 'results/SOTA/IDCN_RGB/LIVE1/10/',
					# 'results/SOTA/QGAC_RGB/LIVE1/jpg10/', 'results/SOTA/FBCNN_RGB/LIVE1/10/', 
					# 'results/SOTA/MARN_3_2_3_64_PR_RGB/LIVE1/jpg10/stage3', 'results/SOTA/MARN_3_2_3_32_PR_RGB/LIVE1/jpg10/stage3']
	imgs_postfix_GT = '.bmp'
	imgs_postfix_Test = ['.jpg']#, '.bmp', '.png', '.png', '.png', '.png']

	img_list = os.listdir(imgs_dir_GT)
	avg_psnr = [0.0] * len(imgs_dir_Test)
	avg_ssim = [0.0] * len(imgs_dir_Test)
	avg_psnrb = [0.0] * len(imgs_dir_Test)
	for img_file in img_list:
		print(img_file)
		img_gt = Image.open(os.path.join(imgs_dir_GT, img_file.split('.')[0] + imgs_postfix_GT))
		img_gt = transforms.ToTensor()(img_gt).unsqueeze(0)
		for i in range(len(imgs_dir_Test)):
			img2 = Image.open(os.path.join(imgs_dir_Test[i], img_file.split('.')[0] + imgs_postfix_Test[i]))
			img2 = transforms.ToTensor()(img2).unsqueeze(0)
			img1 = img_gt
			psnr = calc_psnr(img1, img2)
			psnrb = calc_psnrb(img1, img2)
			ssim = calc_ssim(img1, img2)
			print("%.4f | %.4f | %.6f" % (psnr, psnrb, ssim))
			avg_psnr[i] += psnr
			avg_ssim[i] += ssim
			avg_psnrb[i] += psnrb
	for i in range(len(imgs_dir_Test)):
		avg_psnr[i] /= len(img_list)
		avg_ssim[i] /= len(img_list)
		avg_psnrb[i] /= len(img_list)
		print("PSNR = %.2f, PSNR-B = %.2f, SSIM = %.4f" % (avg_psnr[i], avg_psnrb[i], avg_ssim[i]))
