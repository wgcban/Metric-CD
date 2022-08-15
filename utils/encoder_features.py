import math
import torch
import torchvision
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.transforms import Compose, ToTensor, Normalize

#-------------- VGG Network ----------------#
class VGG_Net(torch.nn.Module):
    def __init__(self, resize=True, requires__grad = True, fine_tune = True):
        super(VGG_Net, self).__init__()
        if fine_tune:
            pretrained = False
        else:
            pretrained = True

        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=pretrained).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=pretrained).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=pretrained).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=pretrained).features[16:23].eval())
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = requires__grad
            
        self.blocks    = torch.nn.ModuleList(blocks)

        if fine_tune:
            #Initialize model with VGG16 finetune on AID dataset
            self.blocks.load_state_dict(torch.load("/media/lidan/ssd/Chaminda/change_detection/VGG_finetune_AID/saved_model/vgg16_AID_049fc.pth"), strict=False)

        self.transform = torch.nn.functional.interpolate
        self.mean      = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1))
        self.std       = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1))
        self.resize    = resize

    def forward(self, input):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
        
        #input = (input-self.mean) / (self.std)
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=True)
   
        out_features = []
        x = input
        for block in self.blocks:
            x = block(x)
            out_features.append(x)
        return out_features

#----------------- Autoencoder --------------------#
class Encoder_Net(nn.Module):
    def __init__(self):
        super(Encoder_Net, self).__init__()
        self.baseWidth   = 12 
        self.cardinality = 8 
        self.scale       = 6 
        self.stride      = 1

        self.encoder_lv1 = nn.Sequential(
            ############# Block1-scale 1.0  ############## 0:1
            nn.Conv2d(3,16,3,1,1),
            Bottle2neckX(16,16, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal'),
        )

        self.encoder_lv2 = nn.Sequential(
            ############# Block2-scale 0.50  ############## 2:4
            TransitionBlock1(32,32),
            Bottle2neckX(32,32, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal'),
            TransitionBlock3(64,32),
        )

        self.encoder_lv3 = nn.Sequential(
            ############# Block3-scale 0.250  ############## 5-7
            TransitionBlock1(32,32),
            Bottle2neckX(32,32, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal'),
            TransitionBlock3(64,64),
        )

        self.encoder_lv4 = nn.Sequential(
            ############# Block4-scale 0.125  ############## 8-10
            TransitionBlock1(64,128),
            Bottle2neckX(128,128, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal'),
            TransitionBlock3(256,128),
        
            ############# Block5-scale 0.125  ############## 11-12
            Bottle2neckX(128,128, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal'),
            TransitionBlock3(256,128),

            ############# Block6-scale 0.125  ############## 13-14
            Bottle2neckX(128,128, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal'),
            TransitionBlock3(256,128)    
        )
        
    def forward(self, xin):
        transform_input = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        xin = transform_input(xin)
        
        output = []
        # Encoder
        x1 = self.encoder_lv1(xin)
        x2 = self.encoder_lv2(x1)
        x3 = self.encoder_lv3(x2)
        x4 = self.encoder_lv4(x3)
        
        # Output
        output.append(x1)
        output.append(x2)
        output.append(x3)
        output.append(x4)
       
        return output

    
# Convolutional Blocks
def conv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))

#------ SEBlock -----#
class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

#------ Bottle2neckX -----#
class Bottle2neckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist block of a new stage.
        """
        super(Bottle2neckX, self).__init__()

        D = int(math.floor(planes * (baseWidth/64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C*scale, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(D*C*scale)
        self.SE = SEBlock(inplanes,C)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(D*C, D*C, kernel_size=3, stride = stride, padding=1, groups=C, bias=False))
          bns.append(nn.BatchNorm2d(D*C))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(D*C*scale, inplanes  , kernel_size=1, stride=1, padding=0, bias=False)        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.width  = D*C
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        residual = x

        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          
          sp = self.relu(self.bns[i](sp))
          sp = self.convs[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        #out = self.SE(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #pdb.set_trace()
        out += residual
        return torch.cat([x, out], 1)

#------ TransitionBlock -----#
class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

#------ TransitionBlock1 -----#
class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

#------ TransitionBlock2 -----#
class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

#------ TransitionBlock -----#
class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)

#------ TransitionBlock2 -----#
class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

#------ TransitionBlock3 -----#
class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out