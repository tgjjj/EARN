import os
from PIL import Image

input_dir = './bmp'
quality = 90
output_dir = './jpg' + str(quality)
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

img_list = os.listdir(input_dir)
for img_file in img_list:
	print(img_file)
	img = Image.open(os.path.join(input_dir, img_file))
	img.save(os.path.join(output_dir, img_file.split('.')[0] + '.jpg'), quality = quality)