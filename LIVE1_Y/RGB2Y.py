import os
from PIL import Image
from torchvision import transforms

def RGB2Y_new(img):
        r = img[:,0,:,:]
        g = img[:,1,:,:]
        b = img[:,2,:,:]
        out = (r * 65.481 + g * 128.553 + b * 24.966 + 16.0) / 255.0
        out = ((out * 255).round()) / 255.0
        return out

def RGB2Y_QGAC(tensor):
        tensor = tensor.squeeze(0)
        tensor = tensor * 255
        tensor = 16. + tensor[0, :, :] * 65.481 / 255. + tensor[1, :, :] * 128.553 / 255. + tensor[2, :, :] * 24.966 / 255.
        tensor = tensor.unsqueeze(0).round() / 255.
        return tensor

input_dir = '../LIVE1/bmp/'
output_dir = './png_test'
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

img_list = os.listdir(input_dir)
for img_file in img_list:
	print(img_file)
	img = Image.open(os.path.join(input_dir, img_file))
	# img = img.convert('L')
	t = transforms.ToTensor()(img).unsqueeze(0)
	t = RGB2Y_new(t)
	img = transforms.ToPILImage()(t)
	img.save(os.path.join(output_dir, img_file.split('.')[0] + '.png'))
