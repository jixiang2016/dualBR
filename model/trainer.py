import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import datetime
import os
from model.dualBR import dualBR
from model.loss import *
from utils.distributed_utils import (broadcast_scalar, is_main_process,
                                            reduce_dict, synchronize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, config,local_rank=-1):
        self.local_rank = local_rank
        self.input_num = config.input_num
        self.output_num = config.output_num
        self.net_model = dualBR(config.input_num, config.output_num)
        self.net_model.to(device)
        if config.training:
            self.optimG = AdamW(self.net_model.parameters(),\
            lr=config.learning_rate, \
            weight_decay=config.weight_decay)
            self.charbonnier = L1_Charbonnier_loss()
        if local_rank != -1:
            self.net_model = DDP(self.net_model, device_ids=[local_rank], output_device=local_rank)
        
    def inference(self, imgs_tensor,encode_map):
        self.net_model.eval()
        # final_out: (batch,3*output_num,h,w)
        flow_list, warped_imgs_list,final_out = self.net_model(imgs_tensor,encode_map,scale=[4, 2, 1],training=False)
        return final_out, flow_list,warped_imgs_list
             
    def update(self, imgs_tensor, encode_map, gts_tensor, learning_rate=0, training=False):
        assert gts_tensor != None, 'gts should not be none!'
        if training:
            for param_group in self.optimG.param_groups:
                param_group['lr'] = learning_rate 
            self.net_model.train()
        else:    # validation 
            self.net_model.eval()
        # final_out: (batch,3*output_num,h,w)
        flow_list, warped_imgs_list,final_out = self.net_model(imgs_tensor,encode_map,scale=[4, 2, 1],training=training)
        
        loss_charb = self.charbonnier(final_out,gts_tensor)
        loss_G = loss_charb 
        
        if training:
            self.optimG.zero_grad()
            loss_G.backward()
            #torch.nn.utils.clip_grad_norm_( self.net_model.parameters() , 1 ) # 3,5,10,20,25,50
            self.optimG.step()
        

        return final_out, {
            'warped': warped_imgs_list[2],
            'flow': flow_list[2],
            'loss_charb': loss_charb,
            'loss_total': loss_G,
            }
            
    
    def save_model(self, args,step,best_dict,update_best):
    
        if not is_main_process():
            return
        dir_name = args.dataset_name+'_'+args.data_mode1+'-'+args.data_mode2+'_'+str(args.input_num)+'_'+str(args.output_num)
        dir_path = os.path.join(args.output_dir,dir_name,'models')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            
        ckpt_filepath = os.path.join(
            dir_path, "model_%d.ckpt" % step
        )
        best_ckpt_filepath = os.path.join(
            args.output_dir,dir_name, "best.ckpt"
        )
        ckpt = {
            "model": self.net_model.state_dict(),
           
        }
        
        ckpt.update(best_dict)
        torch.save(ckpt, ckpt_filepath)
        
        if update_best:
            torch.save(ckpt, best_ckpt_filepath)
        
    def load_model(self, path):
        if device == "cuda":
            ckpt = torch.load(path, map_location=device)
        else:
            ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        ckpt_model = ckpt["model"]
        
        new_dict = {}
        
        
        for attr in ckpt_model:
            if self.local_rank==-1 and attr.startswith("module."): # non-parallel mode load model trained by parallel mode
                new_dict[attr.replace("module.", "", 1)] = ckpt_model[attr]
            elif self.local_rank >=0 and not attr.startswith("module."):# parallel mode load model trained by non-parallel mode
                new_dict["module." + attr] = ckpt_model[attr]
            else:
                new_dict[attr] = ckpt_model[attr]             
                
        self.net_model.load_state_dict(new_dict)        

        return {
            'best_monitored_value': ckpt['best_monitored_value'],
            'best_psnr':ckpt['best_psnr'],
            'best_ssim':ckpt['best_ssim'],
            'best_monitored_iteration':ckpt['best_monitored_iteration'],
            'best_monitored_epoch':ckpt['best_monitored_epoch'], 
            'best_monitored_epoch_step':ckpt['best_monitored_epoch_step'],
        }           