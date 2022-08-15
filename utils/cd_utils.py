import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

#-----  Reading MSI image of ONERA dataset ------#
def get_onera_img(path1, path2, Nbands):
    #----- Reading Img1 -----#
    img1 = []
    for band in range(Nbands):
        temp = load(path1+'B'+str(band+1).zfill(2)+'.tif')
        img1.append(np.array(temp))
    
    #----- Reading Img2 -----#
    img2 = []
    for band in range(Nbands):
        temp = load(path2+'B'+str(band+1).zfill(2)+'.tif')
        img2.append(np.array(temp))
    
    #----- Converting Img1 and Img2 to numpy -----#
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    
    #----- CC Normalization ------- #
    img1, img2 = CC_Processing(img1, img2)
    
    #img1,cd = histeq(img1)
    #img2,cd = histeq(img2)
    
    #----- Normalizing Img1 and Img 2 -----#
    #img1 = (img1-np.mean(img1))/ np.std(img1)
    #img2 = (img2-np.mean(img2))/ np.std(img2)
    
    #----- Normalizing between 0-1 -----#
    img1 = (img1-np.amin(img1))/(np.amax(img1)-np.amin(img1))
    img2 = (img2-np.amin(img2))/(np.amax(img2)-np.amin(img2))
    return img1, img2

def get_onera_RGBimg(path1, path2):
    #----- Reading Img1 -----#
    img1 = load(path1+'img1.png')
    img2 = load(path2+'img2.png')
    
    #----- Converting Img1 and Img2 to numpy -----#
    img1 = np.transpose(np.array(img1), (2, 0, 1) ).astype(np.float32)
    img2 = np.transpose(np.array(img2), (2, 0, 1) ).astype(np.float32)
    
    #----- CC Normalization ------- #
    img1, img2 = CC_Processing(img1, img2)
    
    #img1,cd = histeq(img1)
    #img2,cd = histeq(img2)
    
    #----- Normalizing Img1 and Img 2 -----#
    #img1 = (img1-np.mean(img1))/ np.std(img1)
    #img2 = (img2-np.mean(img2))/ np.std(img2)
    
    #----- Normalizing between 0-1 -----#
    img1 = (img1-np.amin(img1))/(np.amax(img1)-np.amin(img1))
    img2 = (img2-np.amin(img2))/(np.amax(img2)-np.amin(img2))

    return img1, img2 

def get_SZTAKI_RGBimg(path1, path2):
    #----- Reading Img1 -----#
    img1 = load(path1+'im1.bmp')
    img2 = load(path2+'im2.bmp')
    
    #----- Converting Img1 and Img2 to numpy -----#
    img1 = np.transpose(np.array(img1), (2, 0, 1) ).astype(np.float32)
    img2 = np.transpose(np.array(img2), (2, 0, 1) ).astype(np.float32)
    
    #----- CC Normalization ------- #
    img1, img2 = CC_Processing(img1, img2)
    
    #img1,cd = histeq(img1)
    #img2,cd = histeq(img2)
    
    #----- Normalizing Img1 and Img 2 -----#
    #img1 = (img1-np.mean(img1))/ np.std(img1)
    #img2 = (img2-np.mean(img2))/ np.std(img2)
    
    #----- Normalizing between 0-1 -----#
    img1 = (img1-np.amin(img1))/(np.amax(img1)-np.amin(img1))
    img2 = (img2-np.amin(img2))/(np.amax(img2)-np.amin(img2))

    return img1, img2


def get_QB_img(img_path, IMAGE_N):
    #----- Reading Img1 -----#
    img1 = load(img_path+"/"+str(IMAGE_N)+'/t1.bmp')
    img2 = load(img_path+"/"+str(IMAGE_N)+'/t2.bmp')
    
    #----- Converting Img1 and Img2 to numpy -----#
    img1 = np.transpose(np.array(img1), (2, 0, 1) ).astype(np.float32)
    img2 = np.transpose(np.array(img2), (2, 0, 1) ).astype(np.float32)
    
    #----- CC Normalization ------- #
    img1, img2 = CC_Processing(img1, img2)
    
    #img1,cd = histeq(img1)
    #img2,cd = histeq(img2)
    
    #----- Normalizing Img1 and Img 2 -----#
    #img1 = (img1-np.mean(img1))/ np.std(img1)
    #img2 = (img2-np.mean(img2))/ np.std(img2)
    
    #----- Normalizing between 0-1 -----#
    img1 = (img1-np.amin(img1))/(np.amax(img1)-np.amin(img1))
    img2 = (img2-np.amin(img2))/(np.amax(img2)-np.amin(img2))
    
    cm = load(img_path+"/"+str(IMAGE_N)+'/gt.bmp')
    cm = np.array(cm).astype(np.float32)

    cm = 255.0-cm

    return img1, img2, cm

# Read Ottawa dataset image
def get_ottawa_img(path1, path2):
    img1 = load(path1)
    img2 = load(path2)
    
    img1 = np.transpose(np.array(img1), (2, 0, 1)).astype(np.float32)
    img2 = np.transpose(np.array(img2), (2, 0, 1)).astype(np.float32)
    
    #img1 = (img1-np.mean(img1))/ np.std(img1)
    #img2 = (img2-np.mean(img2))/ np.std(img2)
    
    img1,cd = histeq(img1)
    img2,cd = histeq(img2)
    return img1, img2 

# Read Change Detection Dataset
def get_cdd_img(img_no):
    img1 = load('cd_datasets/CDD/Real/subset/val/A/'+img_no+'.jpg')
    img2 = load('cd_datasets/CDD/Real/subset/val/B/'+img_no+'.jpg')
    
    img1 = np.transpose(np.array(img1), (2, 0, 1)).astype(np.float32)
    img2 = np.transpose(np.array(img2), (2, 0, 1)).astype(np.float32)
    
    img1 = (img1-np.mean(img1))/ np.std(img1)
    img2 = (img2-np.mean(img2))/ np.std(img2)
    
    #img1,cd = histeq(img1)
    #img2,cd = histeq(img2)
    return img1, img2

# Read MERA Masks
def get_onera_cm(path):
    cm = load(path+'cm.png')
    cm = np.array(cm).astype(np.float32)
    return cm

def get_SZTAKI_cm(path):
    cm = load(path+'gt.bmp')
    cm = np.array(cm).astype(np.float32)
    return cm

# Read Change Masks
def get_ottawa_cm(path):
    cm = load(path)
    cm = np.array(cm).astype(np.float32)
    return cm

# Read CDD Masks
def get_cdd_cm(img_no):
    cm = load('cd_datasets/CDD/Real/subset/val/OUT/'+img_no+'.jpg')
    cm = np.array(cm).astype(np.float32)
    return cm

# Image Loss
def image_loss(img1, img2, change, no_change, lossf_s, lossf_d):
    #Removing unnecessary dimensions
    no_change_s = no_change.unsqueeze(0).unsqueeze(0)
    change_s    = change.unsqueeze(0).unsqueeze(0)
    
    #Masking and extracting change probability
    change_s    = change_s*(change_s.detach()>no_change_s.detach())
    no_change_s = no_change_s*(change_s.detach()<no_change_s.detach())

    #Loss similarity
    loss_s  = lossf_s(no_change_s*img1, no_change_s*img2)
    
    #Loss dissimilarity
    loss_d  = -lossf_d(change_s*img1, change_s*img2)
    return loss_s, loss_d

# Feature Loss
def feature_loss(img1_feat, img2_feat, change, no_change, lossf_s, lossf_d, N_vgg):
    loss_s = 0.0
    loss_d = 0.0
   
    for i in range(N_vgg):
        img1 = img1_feat[i]
        img2 = img2_feat[i]
        
        #Normalizing feature maps...
        img1_mean = torch.mean(img1, (2,3), True)
        img2_mean = torch.mean(img2, (2,3), True)
        img1_std = torch.std(img1, (2,3), True).unsqueeze(2).unsqueeze(3)
        img2_std = torch.std(img2, (2,3), True).unsqueeze(2).unsqueeze(3)
        
        #Normalizing
        img1 = (img1-img1_mean)/(img1_std+1e-9)
        img2 = (img2-img2_mean)/(img2_std+1e-9)
        
        #Interpolating the change and unchange probability
        if i==0:
            #Same spatial size: No interpolation required
            no_change_s = no_change.unsqueeze(0).unsqueeze(0)
            change_s    = change.unsqueeze(0).unsqueeze(0)
        elif i != 0:
            #Spatial resolution is different: so we interpolate change map accrdingly
            no_change_s = nn.functional.interpolate(no_change.unsqueeze(0).unsqueeze(0), scale_factor=1/2**i, mode='nearest')
            change_s    = nn.functional.interpolate(change.unsqueeze(0).unsqueeze(0), scale_factor=1/2**i, mode='nearest')
        
        #no_change_s = nn.functional.interpolate(no_change.unsqueeze(0).unsqueeze(0), scale_factor=1/2**i, mode='nearest')
        #change_s    = nn.functional.interpolate(change.unsqueeze(0).unsqueeze(0), scale_factor=1/2**i, mode='nearest')
        
        change_s  = change_s*(change_s.detach()>no_change_s.detach())
        no_change_s = no_change_s*(change_s.detach()<no_change_s.detach())
        
        #Loss similarity
        loss_s = loss_s + lossf_s(no_change_s*img1, no_change_s*img2)
    
        #Loss dissimilarity
        loss_d = loss_d - lossf_d(change_s*img1, change_s*img2)
    
    return loss_s, loss_d

# Feature Loss
def contrastive_loss(img1_feat, img2_feat, distance_measure, N_vgg):
    loss     =  0.0
    for i in range(N_vgg):
        img1 = img1_feat[i]
        img2 = img2_feat[i]
    
        #Loss dissimilarity
        loss = loss + distance_measure(img1, img2)
    return loss



def histeq(im, nbr_bins=256):
    #get image histogram
    imhist, bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    
    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf


#------   CC Normalization   ------#
def CC_Processing(R, T):
    # Transfering R to T: R----->T
    
    #Reshaping matrices to vectors
    [b, w, h] = np.shape(R)
    Rr = np.reshape(R, (b, h*w))
    Tr = np.reshape(T, (b, h*w))
    
    #Calculating mean and covariance matrices
    m_R  = np.mean(Rr, axis = 1, keepdims=True)
    m_T  = np.mean(Tr, axis = 1, keepdims=True)
    C_R  = np.matmul((Rr-m_R), np.transpose(Rr-m_R))
    C_T  = np.matmul((Tr-m_T), np.transpose(Tr-m_T))
    C_TR = np.matmul((Tr-m_T), np.transpose(Rr-m_R))
    
    #Transforming R ----> T
    PR = np.matmul(np.matmul(C_TR, np.linalg.inv(C_R)), (Rr-m_R)) + m_T
    PT = T
    
    #Reshaing to original size
    PR = np.reshape(PR, (b, w, h))
    PT = np.reshape(PT, (b, w, h))
    
    
    return PR, PT


#------ Learnable Distance Metric -----#
class distance_metric(nn.Module):
    def __init__(self, bands, W, H):
        super(distance_metric, self).__init__()
        
        self.bands = bands
        self.W = W
        self.H = H
        
        self.L = torch.nn.Parameter(torch.randn(self.bands, self.bands))
        self.L.requires_grad = True
        self.L2 = torch.nn.MSELoss()

    def forward(self, img1, img2):
        img1 = img1.view(self.bands, self.W*self.H)
        img2 = img2.view(self.bands, self.W*self.H)
        
        d =  self.L2(self.L*img1, self.L*img2)
        
        return self.L, d