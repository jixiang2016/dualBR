import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.warplayer import warp
from model.stn import STN


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel)) #torch.nn.LayerNorm
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,\
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

def warp_images(img, flows,length):
    warp_list = []
    for idx in range(length):
        flow_idx = flows[:, 2*idx:2*(idx+1)]
        warp_list.append(warp(img, flow_idx))
    warped_imgs = torch.cat(warp_list,1)
    return warped_imgs
    
    
class dualBR(nn.Module):
    def __init__(self,input_num,output_num):
        super(dualBR, self).__init__()
        self.input_num = input_num
        self.output_num = output_num         

        self.block0 = MotionIntrpl(3,c=240,out_imgs=self.output_num,\
                              in_planes_s=6)
        self.block1 = MotionIntrpl(3*(1+output_num)+output_num*2, c=150,out_imgs=self.output_num,\
                              in_planes_s=10+output_num)
        self.block2 = MotionIntrpl(3*(1+output_num)+output_num*2, c=90,out_imgs=self.output_num,\
                              in_planes_s=10+output_num)
        #####  GenNet
        self.contextnet = Contextnet()
        self.unet = Unet(input_num, output_num)

    def forward(self, imgs_tensor,temp_map,scale=[4,2,1],training=False):
        # imgs_tensor: blur, rs (batch_size,3*input_num,h,w)
        # gts_tensor: multi gts   (batch_size,3*output_num,h,w)
        # temp_map (b,output_num,h,w)  temporal-order encoding for RS branch
        blur = imgs_tensor[:,:3]  ## blur input 
        rg = imgs_tensor[:,3:]  ## RS input 

        mask_list = [] 
        flow_list = [] 
        warped_imgs_list = [] 
        merged_warped_imgs_list = []

        flow = None 
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:            
                flow_d, mask_d, flow_br_d= stu[i](imgs_tensor,temp_map,warped_imgs,flow, mask, flow_br,scale=scale[i])                  
                flow = flow + flow_d
                mask = mask + mask_d
                flow_br = flow_br + flow_br_d
            else:
                flow, mask, flow_br = stu[i](imgs_tensor,temp_map, None,None,None,None,scale=scale[i])             
            warped_imgs_b = warp_images(blur,flow[:,:2*self.output_num],self.output_num)  
            warped_imgs_r = warp_images(rg,flow[:,2*self.output_num:],self.output_num)  
            warped_imgs = torch.cat([warped_imgs_b, warped_imgs_r], 1)
            warped_imgs_list.append(warped_imgs)
            flow_list.append(flow)
            mask_list.append(torch.sigmoid(mask))
            
        for i in range(3):
            mask_pred = mask_list[i].unsqueeze(2)
            ### (B,output_num,3,h,w)
            warped_img_pred_t2b = warped_imgs_list[i][:,:3*self.output_num].view(mask_pred.shape[0],self.output_num,3,mask_pred.shape[-2],-1)
            warped_img_pred_b2t = warped_imgs_list[i][:,3*self.output_num:].view(mask_pred.shape[0],self.output_num,3,mask_pred.shape[-2],-1)
            merged_warped_imgs = (warped_img_pred_t2b * mask_pred + (1.0-mask_pred)*warped_img_pred_b2t).view(mask_pred.shape[0],self.output_num*3,mask_pred.shape[-2],-1)
            merged_warped_imgs_list.append(merged_warped_imgs)


        ### GenNet        
        c0 = self.contextnet(blur, flow[:,:2*self.output_num])
        c1 = self.contextnet(rg, flow[:,2*self.output_num:])
        # (batch, 3*output_num, h ,w)
        tmp = self.unet(imgs_tensor, warped_imgs, flow, mask, c0, c1)
        res = tmp[:,:3*self.output_num] * 2 - 1 
        final_out = torch.clamp(merged_warped_imgs_list[2] + res, 0, 1)
        return flow_list, warped_imgs_list,final_out

class MotionIntrpl(nn.Module):
    def __init__(self, in_planes, c, out_imgs,in_planes_s):
        super(MotionIntrpl, self).__init__()
        self.output_num = out_imgs
        out_planes = 2*self.output_num
        
        #### contextual branch for blur input
        self.conv0_b = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock_b = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),  #### (c,c/2)
        )
        self.lastconv_b = nn.ConvTranspose2d(2*c+c, out_planes, 4, 2, 1)  ###(c+c/3)
        
        
        #### temporal branch for RS/RSGR input
        self.conv0_r = nn.Sequential(
            conv(in_planes+self.output_num, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock_r = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),  #### (c,c/2)
        )
        self.lastconv_r = nn.ConvTranspose2d(2*c+c, out_planes, 4, 2, 1)  ###(c+c/3)
        
        ### shutter alignment
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_planes_s, 64, 3, 1, 1),
            ResBlock(64, 64),
            ResBlock(64, 64),
            conv(64,128,3,2,1),
            )   
        self.stn_br_1 = STN(in_channels = 128)
        self.stn_rb_1 = STN(in_channels = 128)
        self.refine1 = nn.Sequential(
            conv(128*2,256),
            nn.Conv2d(256, 128, 3, 1, 1)
            ) 
                      
        self.encoder2 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            conv(128,256,3,2,1),
            )
        self.stn_br_2 = STN(in_channels = 256)
        self.stn_rb_2 = STN(in_channels = 256)
        self.refine2 = nn.Sequential(
            conv(128+256*2,256),
            nn.Conv2d(256, 2, 3, 1, 1)
            )  
        
        self.conv_e = nn.Conv2d(256, c, 3, 1, 1)
        self.out = nn.Conv2d(c, 1*out_imgs, 3, 1, 1)
        
        
    def forward(self, imgs_tensor,temp_map, warped_imgs,flow, mask, flow_br,scale):
        #temp_map (b,output_num,h,w)
        height,width = imgs_tensor.shape[-2:]
        
        imgs_tensor = F.interpolate(imgs_tensor, scale_factor = 1. / scale, mode="bilinear", align_corners=False)  # resize to 1/K == 1/4,2,1
        temp_map = F.interpolate(temp_map, scale_factor = 1. / scale, mode="bilinear", align_corners=False) 
        blur = imgs_tensor[:,:3]
        RG = torch.cat([imgs_tensor[:,3:],temp_map],dim=1)
        #temp_map_m = F.interpolate(temp_map, scale_factor = 1. / 2, mode="bilinear", align_corners=False)
        if flow != None:
            warped_imgs = F.interpolate(warped_imgs, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            flow_br = F.interpolate(flow_br, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            
            imgs_tensor = torch.cat((imgs_tensor, flow_br,mask), 1)
            blur = torch.cat((blur, flow[:,:self.output_num*2],warped_imgs[:,:self.output_num*3]), 1)
            RG = torch.cat((RG, flow[:,self.output_num*2:],warped_imgs[:,self.output_num*3:]), 1)
            
        ### shutter alignment
        out1 = self.encoder1(imgs_tensor)##(B,128,h/2,w/2)
        flow_b2r_1 = self.refine1(torch.cat([out1,self.stn_br_1(out1)],dim=1))#(B,128,h/2,w/2)
        flow_b2r_1 = F.interpolate(flow_b2r_1, scale_factor = 1. / 2, mode="bilinear", align_corners=False)
        flow_r2b_1 = self.refine1(torch.cat([out1,self.stn_rb_1(out1)],dim=1))#(B,128,h/2,w/2)
        flow_r2b_1 = F.interpolate(flow_r2b_1, scale_factor = 1. / 2, mode="bilinear", align_corners=False)
        out2 = self.encoder2(out1)##(B,256,h/4,w/4)
        flow_b2r_2 = self.refine2(torch.cat([out2,self.stn_br_2(out2),flow_b2r_1],dim=1))#(B,2,h/4,w/4)
        flow_r2b_2 = self.refine2(torch.cat([out2,self.stn_rb_2(out2),flow_r2b_1],dim=1))#(B,2,h/4,w/4)
        ext_fea = self.conv_e(out2) # ##(B,c//3,h/4,w/4)
        mask = self.out(ext_fea) ###(B,output_num,h/4,w/4)
        mask = F.interpolate(mask, size = [height,width], mode="bilinear", align_corners=False)###(B,output_num,h,w)
        
        ## blur branch        
        x = self.conv0_b(blur)
        x_b = self.convblock_b(x) + x
        
        ## RG branch        
        x = self.conv0_r(RG)
        x_r = self.convblock_r(x) + x ##(B,c,h/4,w/4)
        
        ## blur branch 
        x_b_w = warp(x_r,flow_b2r_2)
        tmp_b = self.lastconv_b(torch.cat([x_b,x_b_w,ext_fea],dim=1))
        #tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        tmp_b = F.interpolate(tmp_b, size = [height,width], mode="bilinear", align_corners=False)
        # flow_t2b: (batch_size,output_num*2,h,w)
        flow_t2b = tmp_b[:,:2*self.output_num] * scale * 2
        
        ## RG branch
        x_r_w = warp(x_b,flow_r2b_2)
        tmp_r = self.lastconv_r(torch.cat([x_r,x_r_w,ext_fea],dim=1))
        #tmp * temp_map_m
        #tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        tmp_r = F.interpolate(tmp_r, size = [height,width], mode="bilinear", align_corners=False)
        # flow_t2r: (batch_size,output_num*2,h,w)
        flow_t2r = tmp_r[:,:2*self.output_num] * scale * 2
        
        flow = torch.cat([flow_t2b,flow_t2r],dim=1)  # [flow_t2b,flow_t2r] (batch_size,output_num*4,h,w)
        
        flow_br = torch.cat([flow_b2r_2,flow_r2b_2],dim=1) # [flow_b2r,flow_r2b] (batch_size,4,h,w)
        flow_br = F.interpolate(flow_br, size = [height,width], mode="bilinear", align_corners=False)
        flow_br = flow_br * scale * 4

        return flow, mask, flow_br

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        c = 16
        self.conv1 = Conv2(3, c//2)
        self.conv2 = Conv2(c//2, c)
        self.conv3 = Conv2(c, 2*c)
        self.conv4 = Conv2(2*c, 4*c)
    
    def forward(self, x, flow):
        #x: tensor(batch_size, 3, h, w)  flow: tensor (batch_size,2*output_num,h,w)
        if flow != None:
            length = flow.shape[1]//2

        x = self.conv1(x)
        if (flow != None):
            size_list = [math.ceil(x/2) for x in flow.shape[-2:]]
            flow = F.interpolate(flow, size=size_list, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            #flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            f1 = warp_images(x,flow,length) # (batch_size,c/2*output_num,h,w)
        else:
            f1 = x.clone()  # (batch_size,c/2,h,w)       
        
        x = self.conv2(x)
        if (flow != None):
            size_list = [math.ceil(x/2) for x in flow.shape[-2:]]
            flow = F.interpolate(flow, size=size_list, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            #flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            f2 = warp_images(x,flow,length)# (batch_size,c*output_num,h,w)
        else:
            f2 = x.clone() # (batch_size,c,h,w) 
        
        x = self.conv3(x)
        if (flow != None):
            size_list = [math.ceil(x/2) for x in flow.shape[-2:]]
            flow = F.interpolate(flow, size=size_list, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            #flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            f3 = warp_images(x,flow,length)# (batch_size,2c*output_num,h,w)
        else:
            f3 = x.clone() # (batch_size,2c,h,w)
                
        x = self.conv4(x)
        if (flow != None):
            size_list = [math.ceil(x/2) for x in flow.shape[-2:]]
            flow = F.interpolate(flow, size=size_list, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            #flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            f4 = warp_images(x,flow,length)# (batch_size,4c*output_num,h,w)
        else:
            f4 = x.clone() # (batch_size,4c,h,w)
        
        return [f1, f2, f3, f4]
    
class Unet(nn.Module):
    def __init__(self,input_num,output_num):
        super(Unet, self).__init__()
        self.input_num = input_num
        self.output_num = output_num
        out = output_num
        c = 16

        self.down0 = Conv2(5*out*2+out+3*input_num, 4*c) #  4c              (c/2*out)*2   
        self.down1 = Conv2(4*c+c*out, 4*c+c*out) #4*c+c*out  (c*out)*2 
        self.down2 = Conv2(4*c+3*c*out, 4*c+3*c*out) #4*c+3*c*out  (2c*out)2   
        self.down3 = Conv2(4*c+7*c*out, 4*c+7*c*out)#4*c+7*c*out  (4c*out)*2
        self.up0 = deconv(4*c+15*c*out, 4*c+3*c*out)#4*c+3*c*out  4*c+3*c*out
        self.up1 = deconv(8*c+6*c*out, 4*c+c*out)#4*c+c*out  4*c+c*out
        self.up2 = deconv(8*c+2*c*out, 4*c)#4*c 4*c
        self.up3 = deconv(8*c, 4*c)
        self.conv = nn.Conv2d(4*c, 3*out, 3, 1, 1)
        
        
    def forward(self,imgs_tensor, warped_imgs, flow, mask,c0, c1):
        # c2: pre c1:nxt
        s0 = self.down0(torch.cat((imgs_tensor, warped_imgs,flow,mask), 1))

        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))

        #!!!
        if x.shape[-2:] != s2.shape[-2:]:
            x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat((x, s2), 1)) 
        #!!!
        if x.shape[-2:] != s1.shape[-2:]:
            x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat((x, s1), 1)) 
        #!!!
        if x.shape[-2:] != s0.shape[-2:]:
            x = F.interpolate(x, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat((x, s0), 1)) 
        
        x = self.conv(x)  ##(batch, 3*output_num, h, w)
        
        return torch.sigmoid(x)
        
        
        
