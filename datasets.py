import glob
import os
import cv2
from torch.utils.data import Dataset
import numpy as np
import random


class CutSwap(object):

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img, img2):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        img_ = img.clone().detach()
        local = []
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            img[:,y1: y2, x1: x2] = img2[:,y1: y2, x1: x2]
            img2[:,y1: y2, x1: x2] = img_[:,y1: y2, x1: x2]
            local.append([y1, y2, x1, x2])
        
        return img,img2,local  
    


class CDRL_Dataset_CutSwap(Dataset):
    def __init__(self, root_path=None, dataset=None, train_val=None, transforms_A=None, transforms_B=None, n_holes=None):
        self.transforms_A = transforms_A
        self.transforms_B = transforms_B
        self.train_val = train_val
        self.files = []
        self.n_holes = n_holes
        
        for data in dataset.split(','):
            if data!='':
                self.total_path = os.path.join(root_path, data, train_val)
                self.files += sorted(glob.glob(self.total_path + "/A/*.*")) +\
                              sorted(glob.glob(self.total_path + "/B/*.*"))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_name = self.files[index % len(self.files)].split('/')[-1]
        img_A = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_COLOR)
        img_ori = img_A.copy()
        A2BB2A_path = self.files[index % len(self.files)].split('/'+self.train_val+'/')[0]+'_A2B_B2A/'
        if '/A/' in self.files[index % len(self.files)]:
            img_B = cv2.imread(A2BB2A_path+self.train_val+ '/A/'+img_name, cv2.IMREAD_COLOR)
        elif '/B/' in self.files[index % len(self.files)]:
            img_B = cv2.imread(A2BB2A_path+self.train_val+ '/B/'+img_name, cv2.IMREAD_COLOR)
        
        transformed_A = self.transforms_A(image=img_A)
        transformed_B = self.transforms_B(image=img_B)
        
        img_A = transformed_A["image"]
        img_B = transformed_B["image"]
        
        cutmix_ = CutSwap(n_holes=self.n_holes, length=64)
        img_A_cutmix = img_A.clone().detach()
        img_B_cutmix = img_B.clone().detach()
        img_A_cutmix,img_B_cutmix, local = cutmix_(img_A_cutmix,img_B_cutmix)
        
        return {"A":img_A , "B": img_B, "A_cutmix": img_A_cutmix,"B_cutmix": img_B_cutmix, "local":local}
    


class CDRL_Dataset_test(Dataset):
    def __init__(self, root_path=None, dataset=None, transforms=None):
        self.total_path = os.path.join(root_path, dataset, 'test')
        self.transforms = transforms
        self.files = sorted(glob.glob(self.total_path + "/A/*.*"))
        
    def __getitem__(self, index):
        name = self.files[index % len(self.files)].split('/')[-1]
        
        img_A = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_COLOR)
        img_B = cv2.imread(self.files[index % len(self.files)].replace('/A/','/B/'), cv2.IMREAD_COLOR)
        
        transformed_A = self.transforms(image=img_A)
        transformed_B = self.transforms(image=img_B)
        
        img_A = transformed_A["image"]
        img_B = transformed_B["image"]
        
        return {"A": img_A, "B": img_B, 'NAME': name}

    def __len__(self):
        return len(self.files)

