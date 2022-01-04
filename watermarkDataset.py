import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os
import re

class WatermarkDataset(Dataset):
    """ Reads in the whole watermark dataset """
    
    def __init__(self, img_dir, nb="all", transform=None):
        self.img_dir = img_dir
        self.transform = transform
         
        self.wm = []
        self.nwm = []
        imgnames = os.listdir(img_dir)
        imgnames.sort(key=lambda x: int(re.search(r'\d+', x).group(0)))
        
        if nb == "all":
            self.nb = len(imgnames)
        else:
            self.nb = nb
        
        for img in imgnames[0:self.nb]:
            img_path = img_dir + "/" + img
            if "w" in img:
                self.wm.append(img_path)
            else:
                self.nwm.append(img_path)
        
    def __len__(self):
        return self.nb//2
    
    def __getitem__(self, idx):
        wimgpath = self.wm[idx]
        imgpath = self.nwm[idx]
        wimg = cv2.imread(wimgpath)
        wimg = cv2.cvtColor(wimg, cv2.COLOR_BGR2RGB)
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            transformed_imgs = self.transform(image=wimg, image1=img)
            wimg = transformed_imgs['image']/255
            img = transformed_imgs['image1']/255
        return wimg, img

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
