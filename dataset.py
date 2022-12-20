import os
import random
from PIL import Image, ImageMath
import torch
import torchvision.transforms as transforms

class AR_Dataset(torch.utils.data.Dataset):
	def __init__(self, image_dir, comp_quality = ["40"], phase = "Train", 
		crop_size = 256, samples_per_image = 4, channels = 3):
		super().__init__()
		self.image_dir = image_dir
		self.gt_dir = 'GT'
		self.comp_format = "jpg"
		self.comp_quality = comp_quality
		self.phase = phase
		self.crop_size = crop_size
		self.samples_per_image = samples_per_image
		assert phase == "Train" or phase == "Test" or phase == "Val", "Invalid parameter phase: " + phase
		self.image_list = os.listdir(os.path.join(image_dir, self.gt_dir))
		self.common_transform = transforms.Compose([
			transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels)
			])
		self.random_crop = transforms.RandomCrop(crop_size)

	def __getitem__(self, index):
		gt_im = self.common_transform(Image.open(
			os.path.join(self.image_dir, self.gt_dir, self.image_list[index])))
		comp_im = Image.open(
			os.path.join(self.image_dir, self.comp_format + random.choice(self.comp_quality), self.image_list[index].split('.')[0] + '.' + self.comp_format))
		qm = torch.tensor(comp_im.quantization[0], dtype = torch.float32)
		qm = qm / 255.0
		comp_im = self.common_transform(comp_im)
		im = torch.stack((gt_im, comp_im))
		_, c, h, w = im.size()
		if self.phase == 'Train':
			im_crops = []
			for i in range(self.samples_per_image):
				im_crops.append(self.random_crop(im))
			im = torch.stack(im_crops, dim = 1)
			qm = qm.unsqueeze(0).expand(self.samples_per_image, qm.size(0))
		elif self.phase == 'Val':
			im = transforms.functional.crop(im, 0, 0, self.crop_size, self.crop_size)
		else:
			im = transforms.functional.crop(im, 0, 0, h - h % 8, w - w % 8)
		return im[1, ...], im[0, ...], qm, self.image_list[index]

	def __len__(self):
		return len(self.image_list)

if __name__ == '__main__':
	train_set = AR_Dataset(image_dir = './LIVE1', phase = "Train", comp_quality = ["40"])
