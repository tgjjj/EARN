import torch
import torchvision.transforms as transforms
import torchvision
import os
import argparse
import itertools
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import Network
import AR_Dataset
from Test import test_model

def train_model_1iter(model, data, optimizer, Loss = None):
	comp_data, gt_data, qt, name = data
	comp_data = comp_data.view(-1, comp_data.size(-3), comp_data.size(-2), comp_data.size(-1)).cuda(device = 0)
	gt_data = gt_data.view(-1, gt_data.size(-3), gt_data.size(-2), gt_data.size(-1)).cuda(device = 0)
	qt = qt.view(-1, qt.size(-1)).cuda(device = 0)
	model_output = model(comp_data, qt)
	loss = torch.zeros(1).cuda(device = 0)
	if Loss is not None:
		if isinstance(model_output, tuple):
			for i in range(len(model_output) - 1):
				loss += Loss(model_output[i], model_output[i + 1].clone().detach())
			loss += Loss(model_output[-1], gt_data)
			loss /= len(model_output)
		else:
			loss += Loss(model_output, gt_data)
	loss.backward()
	return loss.item()

def train_model(args, model, train_data_loader, val_data_loader, optimizer, scheduler, last_epoch = 0):
	writer = SummaryWriter()
	MSE_loss = torch.nn.MSELoss()
	loss_count = 0
	optimizer.zero_grad()
	acp, acs, acpb, arp, ars, arpb = test_model(model.module if args.multi_gpu else model, val_data_loader)
	print("Validate PSNR = " + str(arp) + " / " + str(acp))
	print("Validate PSNR-B = " + str(arpb) + " / " + str(acpb))
	print("Validate SSIM = " + str(ars) + " / " + str(acs))
	for epoch in range(last_epoch + 1, 400):
		print("Epoch " + str(epoch) + ", LR = %.6f" % optimizer.param_groups[0]['lr'])
		losses = 0.0
		count = 0
		with tqdm(train_data_loader) as tdl:
			for data in tdl:
				tdl.set_description("Train Epoch %d" % epoch)
				losses_1iter = train_model_1iter(model, data, optimizer, MSE_loss)
				losses += losses_1iter
				tdl.set_postfix(Loss = losses_1iter, str = 'h')
				count += 1
				if count == args.accumulate_batch:
					optimizer.step()
					optimizer.zero_grad()
					count = 0
		print("Loss = " + str(losses / len(train_data_loader)))
		writer.add_scalar("scalar/Loss", losses / len(train_data_loader), loss_count)
		loss_count += 1
		if epoch % 1 == 0:
			acp, acs, acpb, arp, ars, arpb = test_model(model.module if args.multi_gpu else model, val_data_loader)
			writer.add_scalars("scalar/PSNR", {"Comp" : acp, "Rec" : arp}, epoch)
			writer.add_scalars("scalar/PSNR-B", {"Comp" : acpb, "Rec" : arpb}, epoch)
			writer.add_scalars("scalar/SSIM", {"Comp" : acs, "Rec" : ars}, epoch)
			print("Validate PSNR = " + str(arp) + " / " + str(acp))
			print("Validate PSNR-B = " + str(arpb) + " / " + str(acpb))
			print("Validate SSIM = " + str(ars) + " / " + str(acs))
		scheduler.step()
		checkpoint = {'Model' : model.module if args.multi_gpu else model,  
					  'Optimizer' : optimizer, 
					  'Scheduler' : scheduler, 
					  'Epoch' : epoch}
		if epoch % 2 == 0:
			torch.save(checkpoint, './checkpoint_%d.pth' % epoch)
		torch.save(checkpoint, './latest_checkpoint.pth')

if __name__ == "__main__":
	parser = argparse.ArgumentParser("ARNet train parser")
	parser.add_argument("--resume", default=False, action="store_true", help="resume training")
	parser.add_argument("--multi_gpu", default=False, action="store_true", help="using DataParallel for multi-gpu training")
	parser.add_argument("--accumulate_batch", default=1, type=int, help="backward several batches then update weights")
	args = parser.parse_args()

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

	torch.set_num_threads(4)
	torch.set_num_interop_threads(4)
	tf.config.threading.set_inter_op_parallelism_threads(4)
	tf.config.threading.set_intra_op_parallelism_threads(4)

	train_set = AR_Dataset.AR_Dataset(image_dir = './Flickr2K', phase = "Train", 
                comp_quality = ["10", "20", "30", "40", "50", "60", "70", "80", "90"], 
                original_format = "png", crop_size = 256, samples_per_image = 4, channels = 3)
	train_data_loader = torch.utils.data.DataLoader(train_set, batch_size = 1, shuffle = True, num_workers = 1)
	val_set = AR_Dataset.AR_Dataset(image_dir = 'LIVE1', phase = "Test", 
		comp_quality = ["20"], original_format = "bmp", crop_size = 512, channels = 3)
	val_data_loader = torch.utils.data.DataLoader(val_set, batch_size = 1, shuffle = True, num_workers = 1)

	model = Network.ARNet_3S_2N(in_out_plains = 3, base_plains = 64).cuda(device = 0)
	if args.multi_gpu:
		model = torch.nn.DataParallel(model, device_ids = [0, 1])
	optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0, lr = 3e-4)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, range(360, 360 + 3 * 15, 15), gamma=0.3, last_epoch=-1, verbose=False)
	last_epoch = 0
	if args.resume:
		checkpoint = torch.load('./latest_checkpoint.pth')
		model = checkpoint['Model']
		if args.multi_gpu:
			model = torch.nn.DataParallel(model, device_ids = [0, 1])
		optimizer = checkpoint['Optimizer']
		scheduler = checkpoint['Scheduler']
		last_epoch = checkpoint['Epoch']
	train_model(args, model, train_data_loader, val_data_loader, optimizer, scheduler, last_epoch)

	torch.save(model.module if args.multi_gpu else model, './model.pth')
