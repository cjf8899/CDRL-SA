import argparse
import os
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import *
from datasets import *
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from models import SwinTransformerSys

import cv2
import glob

from metric import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="./datasets/", help="root path")
parser.add_argument("--dataset_name", type=str, default="LEVIR-CD", help="name of the dataset")
parser.add_argument("--save_name", type=str, default="levir", help="name of the dataset")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
opt = parser.parse_args()
print(opt)

os.makedirs('pixel_img/'+opt.save_name, exist_ok=True)
os.makedirs('gener_img/'+opt.save_name, exist_ok=True)

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



generator.load_state_dict(torch.load("saved_models/"+opt.save_name+"/generator_9.pth"))

transforms_ = A.Compose([
    A.Resize(opt.img_height, opt.img_width),
    A.Normalize(), 
    ToTensorV2()
])

val_dataloader = DataLoader(
    CDRL_Dataset_test(opt.root_path, dataset=opt.dataset_name, transforms=transforms_),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    
def pixel_visual(gener_output_, A_ori_, name):
    gener_output = gener_output_.cpu().clone().detach().squeeze()
    A_ori = A_ori_.cpu().clone().detach().squeeze()
    
    pixel_loss = to_pil_image(torch.abs(gener_output-A_ori))
    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()])
    pixel_loss = trans(pixel_loss)

    thre_num= 0.7
    threshold = nn.Threshold(thre_num, 0.)
    pixel_loss = threshold(pixel_loss)
    save_image(pixel_loss, 'pixel_img/'+opt.save_name+'/'+str(name[0]))
    save_image(gener_output.flip(-3), 'gener_img/'+opt.save_name+'/'+str(name[0]), normalize=True)


prev_time = time.time()

loss_G_total = 0

generator.eval()


with torch.no_grad():
    for i, batch in enumerate(val_dataloader):

        img_A = Variable(batch["A"].type(Tensor))
        img_B = Variable(batch["B"].type(Tensor))
        name = batch["NAME"]

        valid = Variable(Tensor(np.ones((img_A.size(0), *patch))), requires_grad=False)
        
        img_A = img_A.cuda()
        img_B = img_B.cuda()
        img_AB = torch.cat([img_A,img_B], dim=1) 
        gener_output = generator(img_AB)

        pixel_visual(gener_output, img_A, name)
            

        loss_pixel = criterion_pixelwise(gener_output, img_B)

        loss_G = lambda_pixel * loss_pixel

        loss_G_total += loss_G
        
    print('----------------------------total------------------------------')
    print('loss_G_total : ', round((loss_G_total/len(val_dataloader)).item(),4))
    


paths = glob.glob('./pixel_img/'+opt.save_name+'/*')

if not os.path.isdir('./pixel_img_morpho'):
    os.mkdir('pixel_img_morpho')
if not os.path.isdir('./pixel_img_morpho/'+opt.save_name):
    os.mkdir('pixel_img_morpho/'+opt.save_name)

for path in paths:
    
    img = cv2.imread(path)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img = cv2.dilate(img, k)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img = cv2.erode(img, k)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    img = cv2.erode(img, k)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img = cv2.erode(img, k)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img = cv2.erode(img, k)


    img_name = path.split('/')[-1]
    cv2.imwrite('./pixel_img_morpho/'+opt.save_name+'/'+img_name, img)



con = ConfuseMatrixMeter(2)
pred_path = glob.glob('./pixel_img_morpho/'+opt.save_name+'/*')

scores_dict = 0.
c = 0

for img_path in tqdm(pred_path):
    gt = cv2.imread(opt.root_path + opt.dataset_name + '/test/label/' + img_path.split('/')[-1].replace('jpg','png'),0)
    gt = cv2.resize(gt,(256,256))
    gt = np.expand_dims(gt,axis=0)
    
    pr = np.expand_dims(cv2.imread(img_path,0),axis=0)

    gt[gt>0] = 1
    pr[pr>0] = 1
    gt = gt.astype(int)
    pr = pr.astype(int)
    
    scores_dict += con.update_cm(gt, pr)

    

scores_dict = (scores_dict/len(pred_path)).astype(int)
scores_dict = con.get_scores(scores_dict)

[print(a,' : ', scores_dict[a]) for a in scores_dict]