import argparse
import os
import numpy as np
import time
import datetime
import sys

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from models import SwinTransformerSys

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--root_path", type=str, default="./datasets/", help="root path")
parser.add_argument("--dataset_name", type=str, default="LEVIR-CD", help="name of the dataset")
parser.add_argument("--save_name", type=str, default="levir", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between sampling of images from generators")
parser.add_argument("--n_holes", type=int, default=2, help="size of the batches")

opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.save_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.save_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

lambda_pixel = 100

patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

generator = SwinTransformerSys(img_size=256,
                                patch_size=4,
                                in_chans=6,
                                num_classes=3,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=8,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()


generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


transforms_aug = A.Compose([
    A.Resize(opt.img_height, opt.img_width),
#     A.ColorJitter(p=0.5), 
    A.Normalize(), 
    ToTensorV2()
])

transforms_ori = A.Compose([
    A.Resize(opt.img_height, opt.img_width),
    A.Normalize(), 
    ToTensorV2()
])


dataloader = DataLoader(
    CDRL_Dataset_CutSwap(root_path=opt.root_path, dataset=opt.dataset_name, train_val='train', 
                  transforms_A=transforms_aug, transforms_B=transforms_ori, n_holes=opt.n_holes),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    CDRL_Dataset_CutSwap(root_path=opt.root_path, dataset=opt.dataset_name,  train_val='train', 
                  transforms_A=transforms_aug, transforms_B=transforms_ori, n_holes=opt.n_holes),
    batch_size=10,
    shuffle=False,
    num_workers=1,
)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    imgs = next(iter(val_dataloader))
    img_A = Variable(imgs["A"].type(Tensor)).cuda()
    img_B = Variable(imgs["B"].type(Tensor)).cuda()
    img_A_cutmix = Variable(imgs["A_cutmix"].type(Tensor)).cuda()
    img_B_cutmix = Variable(imgs["B_cutmix"].type(Tensor)).cuda()
    
    img_AB = torch.cat([img_A_cutmix,img_B], dim=1) 
    img_B_fake = generator(img_AB)
    img_A = img_A[:, [2,1,0],:,:]
    img_A_cutmix = img_A_cutmix[:, [2,1,0],:,:]
    img_B_cutmix = img_B_cutmix[:, [2,1,0],:,:]
    img_B_fake = img_B_fake[:, [2,1,0],:,:]
    img_B = img_B[:, [2,1,0],:,:]
    img_sample = torch.cat((img_A.data, img_A_cutmix.data, img_B_fake.data, img_B.data,img_B_cutmix.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.save_name, batches_done), nrow=5, normalize=True)


prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        img_A = Variable(batch["A"].type(Tensor))
        img_B = Variable(batch["B"].type(Tensor))
        img_A_cutmix = Variable(batch["A_cutmix"].type(Tensor))
        img_B_cutmix = Variable(batch["B_cutmix"].type(Tensor))
        local = batch["local"]
        

        valid = Variable(Tensor(np.ones((img_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((img_A.size(0), *patch))), requires_grad=False)

        # Generator
        optimizer_G.zero_grad()

        img_A = img_A.cuda()
        img_B = img_B.cuda()
        img_A_cutmix = img_A_cutmix.cuda()
        img_B_cutmix = img_B_cutmix.cuda()
        img_AB = torch.cat([img_A_cutmix,img_B_cutmix], dim=1) 
        
        gener_output = generator(img_AB)
        
        gener_output_pred = discriminator(gener_output, img_A)
        
        loss_GAN = criterion_GAN(gener_output_pred, valid)  
        
        loss_pixel = criterion_pixelwise(gener_output, img_A)
        loss_pixel_cutmix = 0
        for lo in local:
            y1, y2, x1, x2 = lo
            loss_pixel_cutmix += criterion_pixelwise(
                gener_output[:,:,y1.item(): y2.item(), x1.item(): x2.item()], 
                img_A[:,:,y1.item(): y2.item(), x1.item(): x2.item()])
        loss_pixel_cutmix = loss_pixel_cutmix/len(local)
        
        loss_pixel = (loss_pixel+loss_pixel_cutmix)/2
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()
        
        
        # Discriminator
        optimizer_D.zero_grad()

        pred_real = discriminator(img_B, img_A)
        loss_real = criterion_GAN(pred_real, valid)

        B_pred_fake = discriminator(gener_output.detach(), img_A)
        loss_fake = criterion_GAN(B_pred_fake, fake)

        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()


        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.save_name, epoch))
    torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.save_name, epoch))






