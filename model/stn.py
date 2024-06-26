""" A plug and play Spatial Transformer Module in Pytorch """ 
import os 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class STN(nn.Module):
    """
    Implements a spatial transformer 
    as proposed in the Jaderberg paper. 
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator 
    3. A roi pooled module.

    The current implementation uses a very small convolutional net with 
    3 convolutional layers and 2 fully connected layers. Backends 
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map. 
    """
    def __init__(self, in_channels, mid_channels=128, kernel_size=3,use_dropout=True):
        super(STN,self).__init__()
        #self._h, self._w = spatial_dims ## feature height and width
        self._in_ch = in_channels 
        self._ksize = kernel_size
        self.dropout = use_dropout
        c = mid_channels
        # localization net 
        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=self._ksize, stride=1, padding=1, bias=False) # size : [1x3x32x32]
        self.conv2 = nn.Conv2d(c, c, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(c, c, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.global_pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(c*4*4, 1024)   #!!!!!!!!!!!!!!!!!
        self.fc2 = nn.Linear(1024, 6)


    def forward(self, x): 
        """
        Forward pass of the STN module. 
        x -> input feature map 
        """
        batch,_,self._h, self._w = x.shape
        batch_images = x
        x = F.relu(self.conv1(x))
        #x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2)
        x = self.global_pool(x)

        x = x.view(batch,-1)
        if self.dropout:
            x = F.dropout(self.fc1(x), p=0.5)
            x = F.dropout(self.fc2(x), p=0.5)
        else:
            x = self.fc1(x)
            x = self.fc2(x) # params [Nx6]
        
        x = x.view(-1, 2,3) # change it to the 2x3 matrix 
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        assert(affine_grid_points.size(0) == batch_images.size(0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(batch_images, affine_grid_points)
        #return rois, affine_grid_points
        return rois

        

        

if __name__ == '__main__':
    x = torch.rand(2, 62, 127, 127)
    model = STN(in_channels = 62)
    y = model(x)
    print(model)
    print(model(x))