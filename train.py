import torch
import torchvision.transforms as transforms
import torchvision
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models
import dataset
from val import val_EARN

def train_1iter(args, model, data, optimizer, Loss):
	comp_data, gt_data, qm, file_name = data
	comp_data = comp_data.view(-1, comp_data.size(-3), comp_data.size(-2), comp_data.size(-1)).cuda(device=args.gpus[0])
	gt_data = gt_data.view(-1, gt_data.size(-3), gt_data.size(-2), gt_data.size(-1)).cuda(device=args.gpus[0])
	qm = qm.view(-1, qm.size(-1)).cuda(device=args.gpus[0])
	output = model(comp_data, qm)
	loss = torch.zeros(1).cuda(device=args.gpus[0])
	for i in range(len(output) - 1):
		loss += Loss(output[i], output[i + 1].clone().detach()) * 1
	loss += Loss(output[-1], gt_data)
	loss.backward()
	return loss.item()

def train_EARN(args, model, train_data_loader, val_data_loader, optimizer, scheduler, last_epoch = 0):
	writer = SummaryWriter()
	MSE_loss = torch.nn.MSELoss()
	loss_count = 0
	optimizer.zero_grad()
	acp, acs, acpb, arp, ars, arpb = val_EARN(args, model.module if args.multi_gpu else model, val_data_loader)
	print("Validate PSNR = " + str(arp) + " / " + str(acp))
	print("Validate PSNR-B = " + str(arpb) + " / " + str(acpb))
	print("Validate SSIM = " + str(ars) + " / " + str(acs))
	max_psnr = arp
	for epoch in range(last_epoch + 1, 260):
		print("Epoch " + str(epoch) + ", LR = %.6f" % optimizer.param_groups[0]['lr'])
		losses = 0.0
		count = 0
		with tqdm(train_data_loader) as tdl:
			for data in tdl:
				tdl.set_description("Train Epoch %d" % epoch)
				losses_1iter = train_1iter(args, model, data, optimizer, MSE_loss)
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
		acp, acs, acpb, arp, ars, arpb = val_EARN(args, model.module if args.multi_gpu else model, val_data_loader)
		writer.add_scalars("scalar/PSNR", {"Comp" : acp, "Rec" : arp}, epoch)
		writer.add_scalars("scalar/PSNR-B", {"Comp" : acpb, "Rec" : arpb}, epoch)
		writer.add_scalars("scalar/SSIM", {"Comp" : acs, "Rec" : ars}, epoch)
		print("Validate PSNR = " + str(arp) + " / " + str(acp))
		print("Validate PSNR-B = " + str(arpb) + " / " + str(acpb))
		print("Validate SSIM = " + str(ars) + " / " + str(acs))
		scheduler.step()
		if arp > max_psnr:
			torch.save(model.module if args.multi_gpu else model, os.path.join(args.exp_path, 'best.pth'))
			max_psnr = arp
		checkpoint = {'Model' : model.module if args.multi_gpu else model,  
					  'Optimizer' : optimizer, 
					  'Scheduler' : scheduler, 
					  'Epoch' : epoch}
		torch.save(checkpoint, os.path.join(args.exp_path, 'latest_checkpoint.pth'))

def increment_path(root, prefix='exp', figures=2):
	for i in range(10 ** figures):
		path = os.path.join(root, prefix + '_%0{}d'.format(figures) % i)
		if not os.path.exists(path):
			os.makedirs(path)
			return path
	print("No empty exp path exists.")
	exit(0)

if __name__ == "__main__":
	parser = argparse.ArgumentParser("EARN train parser")
	# general
	parser.add_argument("--channels", default=3, type=int, help="3 for RGB and 1 for gray scale images")
	parser.add_argument("--resume", default=None, type=str, help="resume training for the designated experiment, for example --resume ./exp/train/exp_00")
	parser.add_argument("--gpus", nargs="+", default=[0], type=int, help="GPUs for training, DataParallel is used for multi-gpu training")
	parser.add_argument("--num_workers", default=1, type=int, help="number of workers for dataloader")
	parser.add_argument("--network_channels", default=64, type=int, help="the output channels of the Encoder, 64 for EARN and 32 for EARN-Lite")
	parser.add_argument("--exp_root", default="./exp/train/", type=str, help="root directory for experiments")
	# train
	parser.add_argument("--train_data", type=str, required=True, help="path to training dataset")
	parser.add_argument("--train_quality", nargs="+", default=[10,20,30,40,50,60,70,80,90], type=int, help="JPEG compression qualities for training")
	parser.add_argument("--train_crop_size", default=256, type=int, help="patch size for training")
	parser.add_argument("--batch_size", default=4, type=int, help="batch size for training, a batch is extracted by randomly sampling from one image")
	parser.add_argument("--accumulate_batch", default=1, type=int, help="backward several batches then update weights")
	parser.add_argument("--epoch", default=400, type=int, help="total epochs for training")
	parser.add_argument("--init_lr", default=3e-4, type=float, help="initial learning rate")
	parser.add_argument("--decay_rate", default=0.3, type=float, help="learning rate is multiplied by this at decay point")
	parser.add_argument("--decay_point", nargs="+", default=[360,375,390], type=int, help="epochs when learning rate need to decay")
	# val
	parser.add_argument("--val_data", type=str, required=True, help="path to validation dataset")
	parser.add_argument("--val_quality", default=20, type=int, help="JPEG compression qualities for validation")
	parser.add_argument("--val_crop_size", default=512, type=int, help="patch size for validation")
	args = parser.parse_args()
	args.multi_gpu = len(args.gpus) > 1
	args.exp_path = increment_path(args.exp_root) if args.resume is None else args.resume

	torch.set_num_threads(4)
	torch.set_num_interop_threads(4)

	train_qualities = [str(x) for x in args.train_quality]
	train_set = dataset.AR_Dataset(image_dir = args.train_data, phase = "Train", 
				comp_quality = train_qualities, crop_size = args.train_crop_size, samples_per_image = args.batch_size, channels = args.channels)
	train_data_loader = torch.utils.data.DataLoader(train_set, batch_size = 1, shuffle = True, num_workers = args.num_workers)
	val_set = dataset.AR_Dataset(image_dir = args.val_data, phase = "Val", 
				comp_quality = [str(args.val_quality)], crop_size = args.val_crop_size, channels = args.channels)
	val_data_loader = torch.utils.data.DataLoader(val_set, batch_size = 1, shuffle = True, num_workers = args.num_workers)

	model = models.EARN(in_out_channels = args.channels, base_channels = args.network_channels).cuda(device = args.gpus[0])
	if args.multi_gpu:
		model = torch.nn.DataParallel(model, device_ids = args.gpus)
	optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0, lr = args.init_lr)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_point, gamma=args.decay_rate, last_epoch=-1, verbose=False)
	last_epoch = 0
	if args.resume is not None:
		checkpoint = torch.load(os.path.join(args.exp_path, 'latest_checkpoint.pth'))
		model = checkpoint['Model']
		if args.multi_gpu:
			model = torch.nn.DataParallel(model, device_ids = args.gpus)
		optimizer = checkpoint['Optimizer']
		scheduler = checkpoint['Scheduler']
		last_epoch = checkpoint['Epoch']
	train_EARN(args, model, train_data_loader, val_data_loader, optimizer, scheduler, last_epoch)
	torch.save(model.module if args.multi_gpu else model, os.path.join(args.exp_path, 'last.pth'))
