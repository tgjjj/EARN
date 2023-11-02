import os
import time
from functools import wraps
from math import ceil

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import Network

def timer(f):
	@wraps(f)
	def decorated(*args, **kwargs):
		start = time.time()
		res = f(*args, **kwargs)
		end = time.time()
		print(f.__name__.ljust(15) + '|' + ('%.2f ms' % ((end - start) * 1000)).rjust(15))
		return res 

	return decorated

# @timer
def inference(ims, qt, net, stage):
	results = []
	with torch.no_grad():
		for im in ims:
			results.append(net(im, qt, test_mode = True, stage_num = stage))
	return results

# @timer
def load_img(path):
	img = Image.open(path)
	a = img.getpixel((0, 0))
	return img

# @timer
def preprocess(img, w, h, channels, crop_size = 1024, stitch_size = 512):
	im = transforms.functional.to_tensor(img).unsqueeze(0).cuda()
	im = transforms.functional.normalize(im, [0.5] * channels, [0.5] * channels)
	stitch_coords = []
	ims = []
	if w * h > crop_size * crop_size:
		assert (crop_size - stitch_size) % 2 == 0, "Unbalanced padding for crop_size = %d and stitch_size = %d." % (crop_size, stitch_size)
		assert stitch_size <= w and stitch_size <= h, "Width or Height is smaller than stitch_size."
		pad_size = (crop_size - stitch_size) // 2
		im = transforms.functional.pad(im, [pad_size, pad_size, pad_size, pad_size], padding_mode = 'constant')
		for x in range(0, w, stitch_size):
			for y in range(0, h, stitch_size):
				x = min(w - stitch_size, x)
				y = min(h - stitch_size, y)
				stitch_coords.append([x, y])
				ims.append(transforms.functional.crop(im, y, x, crop_size, crop_size))
	else:
		# im = transforms.functional.crop(im, 0, 0, im.size(2) - im.size(2) % 8, im.size(3) - im.size(3) % 8)
		im = transforms.functional.pad(im, [0, 0, ceil(w / 8) * 8 - w, ceil(h / 8) * 8 - h], padding_mode = 'edge')
		# print(im.size())
		ims.append(im)
	qt = torch.tensor(img.quantization[0], dtype = torch.float32).unsqueeze(0).cuda()
	qt = qt / 255.0
	return ims, qt, stitch_coords

# @timer
def save_img(img, path):
	img.save(path)

# @timer
def postprocess(results, w, h, channels, stitch_coords, crop_size = 1024, stitch_size = 512):
	if len(stitch_coords) != 0:
		img = torch.zeros((channels, h, w))
		for i, stitch_coord in enumerate(stitch_coords):
			x, y = stitch_coord
			pad_size = (crop_size - stitch_size) // 2
			result = results[i]
			if isinstance(result, tuple):
				result = result[-1]
			img[:, y:y+stitch_size, x:x+stitch_size] = result[0, :, pad_size:pad_size+stitch_size, pad_size:pad_size+stitch_size]
	else:
		result = results[0]
		if isinstance(result, tuple):
			result = result[-1]
		img = result.detach().cpu().squeeze(0)
		img = transforms.functional.crop(img, 0, 0, h, w)
	img = img * 0.5 + 0.5
	img[img < 0] = 0
	img[img > 1] = 1
	if channels == 1:
		img = img.squeeze(0)
	img = transforms.ToPILImage('RGB' if channels == 3 else 'L')(img)
	return img

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
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	dataset_dir = './LIVE1_Y'
	model_path = './model_Y.pth'
	result_dir = './results/Y'
	test_qualities = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
	gt_format = 'png'
	stages = [3, 2, 1]
	channels = 1
	crop_size = 2048
	stitch_size = 2048

	torch.set_num_threads(4)
	torch.set_num_interop_threads(4)

	model = torch.load(model_path).cuda()
	
	for test_quality in test_qualities:
		input_dir = os.path.join(dataset_dir, 'jpg' + test_quality)
		im_list = os.listdir(input_dir)
		for stage in stages:
			output_dir = os.path.join(result_dir, 'jpg' + str(test_quality), 'stage' + str(stage))
			if not os.path.exists(output_dir):
				os.makedirs(output_dir, exist_ok = True)
				im_list = os.listdir(input_dir)
				for im_file in im_list:
					img = load_img(os.path.join(input_dir, im_file))
					w, h = img.size
					ims, qt, stitch_coords = preprocess(img, w, h, channels, crop_size, stitch_size)
					results = inference(ims, qt, model, stage)
					img = postprocess(results, w, h, channels, stitch_coords, crop_size, stitch_size)
					save_img(img, os.path.join(output_dir, im_file.replace('.jpg', '.png')))
			else:
				print('Path %s already exists, only calculate metrics' % output_dir)
			imgs_dir_GT = os.path.join(dataset_dir, gt_format)
			imgs_dir_Test = output_dir
			img_list = os.listdir(imgs_dir_GT)
			avg_psnr = 0.0
			avg_ssim = 0.0
			avg_psnrb = 0.0
			for img_file in img_list:
				# print(img_file)
				img_gt = Image.open(os.path.join(imgs_dir_GT, img_file.split('.')[0] + '.' + gt_format))
				img_gt = transforms.ToTensor()(img_gt).unsqueeze(0)
				img2 = Image.open(os.path.join(imgs_dir_Test, img_file.split('.')[0] + '.png'))
				img2 = transforms.ToTensor()(img2).unsqueeze(0)
				img1 = img_gt
				psnr = calc_psnr(img1, img2)
				psnrb = calc_psnrb(img1, img2)
				ssim = calc_ssim(img1, img2)
				# print("%.4f | %.4f | %.6f" % (psnr, psnrb, ssim))
				avg_psnr += psnr
				avg_ssim += ssim
				avg_psnrb += psnrb
			avg_psnr /= len(img_list)
			avg_ssim /= len(img_list)
			avg_psnrb /= len(img_list)
			print("Quality = %s, Stage = %d" % (test_quality, stage))
			print("PSNR = %.2f, PSNR-B = %.2f, SSIM = %.4f" % (avg_psnr, avg_psnrb, avg_ssim))

