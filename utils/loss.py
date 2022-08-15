import torch
import torch.nn as nn

class info_loss(nn.Module):
    def __init__(self, temp_c = 1.0, temp_nc = 1.0):
        super(info_loss, self).__init__()
        self.temp_c  = temp_c
        self.temp_nc = temp_nc
    
    def forward(self, img1_c, img2_c, img1_nc, img2_nc):
        #Dissimilarity between changed region
        loss_c  = torch.log( torch.exp(torch.mean(img1_c*img2_c/self.temp_c)) / ( torch.exp(torch.mean(img1_c*img2_c/self.temp_c)) + torch.exp(torch.mean(img1_nc*img2_nc/self.temp_nc))) )
        #Simmilarity between unchanged region
        loss_nc = -torch.log( torch.exp(torch.mean(img1_nc*img2_nc/self.temp_nc)) / ( torch.exp(torch.mean(img1_c*img2_c/self.temp_c)) + torch.exp(torch.mean(img1_nc*img2_nc/self.temp_nc))) )
        
        #Total loss
        loss = loss_c+loss_nc
        return loss

# Cross Feature Information Loss
def cm_InfoNce_loss(change, no_change, info_loss):
    #Itterating over different levels of feature maps from pretrained encoder...
    total_loss = 0.0
    for i in range(1):
        #Interpolating the change and unchange probability
        if i == 0:
            #Same spatial size: No interpolation required
            no_change_s = no_change.unsqueeze(0).unsqueeze(0)
            change_s    = change.unsqueeze(0).unsqueeze(0)
        elif i != 0:
            #Spatial resolution is different: so we interpolate change map accrdingly
            no_change_s = nn.functional.interpolate(no_change.unsqueeze(0).unsqueeze(0), scale_factor=1/2**i, mode='nearest')
            change_s    = nn.functional.interpolate(change.unsqueeze(0).unsqueeze(0), scale_factor=1/2**i, mode='nearest')
        
        #Weight img1 and img2 according to change and no-change probability
        #img1_c  = change_s*img1
        #img2_c  = change_s*img2
        #img1_nc = no_change_s*img1
        #img2_nc = no_change_s*img2
        img1_c  = change_s*(change_s.detach()>no_change_s.detach())
        img2_c  = change_s*(change_s.detach()>no_change_s.detach())
        img1_nc = no_change_s*(change_s.detach()<no_change_s.detach())
        img2_nc = no_change_s*(change_s.detach()<no_change_s.detach())

        #Calculating similarity (between unchanged regions) and dissimilarity (between changed regions) 
        total_loss = total_loss + info_loss(img1_c=img1_c, img2_c=img2_c, img1_nc=img1_nc, img2_nc=img2_nc)
    return total_loss

# Cross Feature Information Loss
def feature_InfoNce_loss(img1_feat, img2_feat, change, no_change, info_loss, N_vgg):
    #Itterating over different levels of feature maps from pretrained encoder...
    total_loss = 0.0
    for i in range(N_vgg):
        img1 = img1_feat[i]
        img2 = img2_feat[i]
        
        #Normalizing feature maps...
        img1_mean = torch.mean(img1, (2,3), True)
        img2_mean = torch.mean(img2, (2,3), True)
        img1_std = torch.std(img1, (2,3), True).unsqueeze(2).unsqueeze(3)
        img2_std = torch.std(img2, (2,3), True).unsqueeze(2).unsqueeze(3)
        
        #Normalizing
        img1 = (img1-img1_mean)/(img1_std+1e-6)
        img2 = (img2-img2_mean)/(img2_std+1e-6)
        
        #Interpolating the change and unchange probability
        if i == 0:
            #Same spatial size: No interpolation required
            no_change_s = no_change.unsqueeze(0).unsqueeze(0)
            change_s    = change.unsqueeze(0).unsqueeze(0)
        elif i != 0:
            #Spatial resolution is different: so we interpolate change map accrdingly
            no_change_s = nn.functional.interpolate(no_change.unsqueeze(0).unsqueeze(0), scale_factor=1/2**i, mode='nearest')
            change_s    = nn.functional.interpolate(change.unsqueeze(0).unsqueeze(0), scale_factor=1/2**i, mode='nearest')
        
        #Weight img1 and img2 according to change and no-change probability
        #img1_c  = change_s*img1
        #img2_c  = change_s*img2
        #img1_nc = no_change_s*img1
        #img2_nc = no_change_s*img2
        img1_c  = change_s*(change_s.detach()>no_change_s.detach())*img1
        img2_c  = change_s*(change_s.detach()>no_change_s.detach())*img2
        img1_nc = no_change_s*(change_s.detach()<no_change_s.detach())*img1
        img2_nc = no_change_s*(change_s.detach()<no_change_s.detach())*img2

        #Calculating similarity (between unchanged regions) and dissimilarity (between changed regions) 
        total_loss = total_loss + info_loss(img1_c=img1_c, img2_c=img2_c, img1_nc=img1_nc, img2_nc=img2_nc)
    return total_loss
