import os
import time
from functools import wraps
from math import ceil

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

@timer
def inference(ims, qt, net, stage):
	results = []
	with torch.no_grad():
		for im in ims:
			results.append(net(im, qt, test_mode = True, stage_num = stage))
	return results

@timer
def load_img(path):
	img = Image.open(path)
	a = img.getpixel((0, 0))
	return img

@timer
def preprocess(img, w, h, crop_size = 1024, stitch_size = 512):
	im = transforms.functional.to_tensor(img).unsqueeze(0).cuda()
	im = transforms.functional.normalize(im, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

@timer
def save_img(img, path):
	img.save(path)

@timer
def postprocess(results, w, h, stitch_coords, crop_size = 1024, stitch_size = 512):
	if len(stitch_coords) != 0:
		img = torch.zeros((3, h, w))
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
	img = transforms.ToPILImage('RGB')(img)
	return img

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	input_dir = './ICB/jpg10_crop/'
	output_dir = './results/Crop_Inference_Test/MARN_3_2_3_64_RGB/ICB/jpg10_crop/stage3/'
	model_path = './models/SOTA/MARN_3_2_3_64_RGB.pth'
	stage = 3
	crop_size = 1024
	stitch_size = 1024

	torch.set_num_threads(4)
	torch.set_num_interop_threads(4)

	model = torch.load(model_path).cuda()
	
	if not os.path.exists(output_dir):
		os.makedirs(output_dir, exist_ok = True)

	im_list = os.listdir(input_dir)

	for im_file in im_list:
		img = load_img(os.path.join(input_dir, im_file))
		w, h = img.size
		# print(w, h)
		ims, qt, stitch_coords = preprocess(img, w, h, crop_size, stitch_size)
		results = inference(ims, qt, model, stage)
		img = postprocess(results, w, h, stitch_coords, crop_size, stitch_size)
		save_img(img, os.path.join(output_dir, im_file.replace('.jpg', '.png')))
