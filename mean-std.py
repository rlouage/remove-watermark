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
            wimg = transformed_imgs['image']
            img = transformed_imgs['image1']
        return wimg, img
        
def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        data = data.to("cuda")
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
    
crop_transform = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.augmentations.transforms.Normalize(mean=(0.6349, 0.5809, 0.5312), std=(0.3283, 0.3288, 0.3534), max_pixel_value=255.0, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
        ],  
        additional_targets={'image1': 'image'}
    )

if __name__ == '__main__':
    # hyperparameters
    lr = 3e-4
    betas = (0.5, 0.999)
    batch_size = 16
    n_cpu = 6
    start_epoch = 160
    epochs = 500
    sample_interval = 100
    checkpoint_interval = 20
    load = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train = WatermarkDataset("data/train", nb="all", transform=crop_transform)
    trainloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    
    print(get_mean_std(trainloader))