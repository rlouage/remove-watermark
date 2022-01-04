import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.utils import save_image

def concat_images(rootdir, rows, cols, w, h):
    """ 
        Concats images from every batch to compare progress in batches 
        Every directory should contain a batch with generated images
        Every directories name should be its batch number (or any name in ascending order)
        Arguments:
            rootdir = root directory that contains directories of batches
            rows    = number of rows in concatenated images
            cols    = number of cols in concatenated images
            w       = width of each subimage in the concatenated images
            h       = height of each subimage in the concatenated images
    """
    dirs = os.listdir(rootdir)
    dirs.sort()
    imgs_per_folder = len(os.listdir(rootdir + "/" + dirs[0]))
    for d in tqdm(dirs):
        # don't generated images if they already exist
        if "jpg" in d:
            continue
        if f'{d}.jpg' in dirs:
            continue
        fd = rootdir + "/" + d
        imgs = [cv2.resize(cv2.imread(fd + "/" + str(i) + ".jpg"), (w, h)) for i in range(imgs_per_folder)]
        imgs_per_row = imgs_per_folder // rows
        row_images = [imgs[i*imgs_per_row:(i+1)*imgs_per_row] for i in range(rows)]
        final_rows = [np.hstack(tuple(img)) for img in row_images]
        final_img = np.vstack(final_rows)
        cv2.imwrite(rootdir + "/" + f'{d}.jpg', final_img)

# helper functions
def val_loss(generator, discriminator, vggloss, valloader, Tensor, kernel_size, unfold, patch, criterion_GAN, criterion_pixelwise, lambda_pixel, lambda_vgg, device):
    """ BATCHSIZE SHOULD BE 1 """
    generator.eval()
    discriminator.eval()
    vggloss.eval()
    
    total_loss_G = torch.zeros([1], dtype=torch.float).to(device)
    total_loss_D = torch.zeros([1], dtype=torch.float).to(device)
    loop = tqdm(valloader)
    for i, batch in enumerate(loop):
        # Model inputs
        real_A = Variable(batch[0].type(Tensor))
        real_B = Variable(batch[1].type(Tensor))
        
        # reshape full images to patches of 256x256
        _,_,w,h = real_A.shape
        real_A = transforms.functional.pad(real_A, [0, 0, int((int(h/kernel_size)+1)*kernel_size)-h, int((int(w/kernel_size)+1)*kernel_size)-w])
        real_A = unfold(real_A).squeeze(0).T.view(-1,3,kernel_size,kernel_size)
        _,_,w,h = real_A.shape
        real_B = transforms.functional.pad(real_B, [0, 0, int((int(h/kernel_size)+1)*kernel_size)-h, int((int(w/kernel_size)+1)*kernel_size)-w])
        real_B = unfold(real_B).squeeze(0).T.view(-1,3,kernel_size,kernel_size)
        

        # split data up in single images to fit in memory
        b = real_A.shape[0]
        t_loss_G = torch.zeros([1], dtype=torch.float).to(device)
        t_loss_D = torch.zeros([1], dtype=torch.float).to(device)
        
        for k in range(b):
            real_A_ = real_A[k].clone().unsqueeze(0).to(device)
            real_B_ = real_B[k].clone().unsqueeze(0).to(device)
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A_.size(0), *patch))), requires_grad=False).to(device)
            fake = Variable(Tensor(np.zeros((real_A_.size(0), *patch))), requires_grad=False).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # Gen
                    fake_B = generator(real_A_)
                    pred_fake = discriminator(fake_B, real_A_)
                    loss_GAN = criterion_GAN(pred_fake, valid)
                    # Pixel-wise loss
                    loss_pixel = criterion_pixelwise(fake_B, real_B_)
                    # Total loss
                    loss_G = loss_GAN + lambda_pixel * loss_pixel + lambda_vgg * vggloss.calc_loss(fake_B, real_B_)
                    t_loss_G += loss_G

                    # Disc
                    pred_real = discriminator(real_B_, real_A_)
                    loss_real = criterion_GAN(pred_real, valid)
                    # Fake loss
                    pred_fake = discriminator(fake_B.detach(), real_A_)
                    loss_fake = criterion_GAN(pred_fake, fake)
                    # Total loss
                    loss_D = 0.5 * (loss_real + loss_fake)
                    t_loss_D += loss_D
                    
        t_loss_G = t_loss_G/b
        t_loss_D = t_loss_D/b
        total_loss_G += t_loss_G
        total_loss_D += t_loss_D
                
    total_loss_G = total_loss_G/len(valloader)
    total_loss_D = total_loss_D/len(valloader)
    
    generator.train()
    discriminator.train()
    vggloss.train()
    return total_loss_G,total_loss_D

def sample_images(batches_done, N, generator, valloader, Tensor, kernel_size, unfold, device):
    """ Save the first N generated images from the dataset """
    generator.eval()
    generator.to("cpu")
    
    imgs = []
    for i, batch in enumerate(valloader):
        if i >= N:
            break
        t = []
        real_A = Variable(batch[0].type(Tensor))
        real_B = Variable(batch[1].type(Tensor))
        t.append(real_A.detach())
        
        _,_,w,h = real_A.shape
        real_A = transforms.functional.pad(real_A, [0, 0, int((int(h/kernel_size)+1)*kernel_size)-h, int((int(w/kernel_size)+1)*kernel_size)-w])
        real_A = unfold(real_A).squeeze(0).T.view(-1,3,kernel_size,kernel_size)
        fake_B = generator(real_A)
        
        new_w = int((int(w/kernel_size)+1)*kernel_size)
        new_h = int((int(h/kernel_size)+1)*kernel_size)
        fake_B = fake_B.permute(1,2,3,0)
        fake_B = fake_B.reshape(1,3*kernel_size*kernel_size,fake_B.shape[3])
        fold = nn.Fold((new_w,new_h), (kernel_size,kernel_size), stride=(kernel_size,kernel_size))
        fake_B = fold(fake_B)
        
        fake_B = fake_B[:, :, 0:w, 0:h]
        t.append(fake_B.detach())
        t.append(real_B.detach())
        imgs.append(tuple(t))
    
    os.mkdir(f"images/watermarkdataset/{batches_done}")
    for j, (a,fb,b) in enumerate(imgs):
        img_sample = torch.cat((a, fb, b), -2)
        save_image(img_sample, f"images/watermarkdataset/{batches_done}/{j}.jpg")
    
    generator.to(device)
    generator.train()
    return imgs
