import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import gc

from model.trainer import Model
from dataset.DualRealDataset import *
from utils.distributed_utils import (broadcast_scalar, is_main_process,
                                            reduce_dict, synchronize)
from model.pytorch_msssim import ssim_matlab
from utils.logger import Logger
from utils.timer import (Timer,Epoch_Timer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=300, type=int) 
parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
parser.add_argument('--batch_size_val', default=16, type=int, help='minibatch size')
parser.add_argument('--local_rank', default=0, type=int, help='local rank')
parser.add_argument('--world_size', default=4, type=int, help='world size')

parser.add_argument('--input_num', default=2, type=int, help='input images number')
parser.add_argument('--input_dir', default='/media/zhongyi/D/data/realBR',  type=str, required=True, help='path to the input dataset folder')
parser.add_argument('--dataset_name', default='realBR',  type=str, required=True, help='Name of dataset to be used')
parser.add_argument('--data_mode1', default='Blur',  type=str, help='Mode of input data')
parser.add_argument('--data_mode2', default='RS',  type=str, help='Mode of input data')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=0 , type=float) 

parser.add_argument('--temporal', action='store_true',default=False)  #### Read consecutive images and taken as input.
parser.add_argument('--training', default=True,  type=bool)
parser.add_argument('--output_num', default=7, type=int, help='final output channel of the network')
parser.add_argument('--output_dir', default='',  type=str, required=True, help='path to save training output')
parser.add_argument('--should_log', default=True,  type=bool)
parser.add_argument('--resume', default=False,  type=bool)
parser.add_argument('--resume_file', default=None,  type=str, help='path to resumed model')

args = parser.parse_args()


# Gradually reduce the learning rate using cosine annealing
def get_learning_rate(step):
    step =step+1
    if step < 3000:
        mul = step / 3000.
        return args.learning_rate * mul
    else:
        mul = np.cos((step - 3000) / (args.epoch * args.step_per_epoch - 3000.) * math.pi) * 0.5 + 0.5
        return (args.learning_rate - 1e-6) * mul + 1e-6

def _summarize_report(prefix="", should_print=True, extra={},log_writer=None,current_iteration=0,max_iterations=0):
        if not is_main_process():
            return
        if not should_print:
            return     
        print_str = []
        if len(prefix):
            print_str += [prefix + ":"]
        print_str += ["{}/{}".format(current_iteration, max_iterations)]
        print_str += ["{}: {}".format(key, value) for key, value in extra.items()]
        log_writer.write(','.join(print_str)) 


def train(model):
    log_writer = Logger(args) 
    log_writer.write("Torch version is: " + torch.__version__)
    log_writer.write("===== Model =====")
    log_writer.write(model.net_model)
    
    #resume 
    if args.resume is True:
        log_writer.write("Restore traing from saved model")
        if args.resume_file is None:
            dir_name = args.dataset_name+'_'+args.data_mode1+'-'+args.data_mode2+'_'+str(args.input_num)+'_'+str(args.output_num)
            checkpoint_path = os.path.join(args.output_dir,dir_name,'best.ckpt')
        else:
            checkpoint_path = args.resume_file
        checkpoint_info = model.load_model(path=checkpoint_path)
       
    if is_main_process():
        writer = SummaryWriter('./tensorboard_log/train')
        writer_val = SummaryWriter('./tensorboard_log/validate')
    else:
        writer = None
        writer_val = None
    
    data_root = os.path.join(args.input_dir, args.dataset_name)                          
    if args.dataset_name == 'realBR': 
        args.inter_num = 16
        args.intra_num = 9      
        dataset = DualRealDataset(dataset_cls='train',\
                               input_num=args.input_num,\
                               output_num=args.output_num,\
                               data_root=data_root,\
                               data_mode1 = args.data_mode1,\
                               data_mode2 = args.data_mode2,\
                               inter_num = args.inter_num,\
                               intra_num = args.intra_num,temp=args.temporal)

        dataset_val = DualRealDataset(dataset_cls='validate',\
                               input_num=args.input_num,\
                               output_num=args.output_num,\
                               data_root=data_root,\
                               data_mode1 = args.data_mode1,\
                               data_mode2 = args.data_mode2,\
                               inter_num = args.inter_num,\
                               intra_num = args.intra_num,temp=args.temporal) 
                               
    elif args.dataset_name == 'GOPRO-VFI_copy': 
        if args.output_num >8:
            raise Exception('Wrong output number!!!')
        args.inter_num = 0
        args.intra_num = 8     
        dataset = DualRealDataset(dataset_cls='train',\
                               input_num=args.input_num,\
                               output_num=args.output_num,\
                               data_root=data_root,\
                               data_mode1 = args.data_mode1,\
                               data_mode2 = args.data_mode2,\
                               inter_num = args.inter_num,\
                               intra_num = args.intra_num,temp=args.temporal)
                               
        dataset_val = DualRealDataset(dataset_cls='test',\
                               input_num=args.input_num,\
                               output_num=args.output_num,\
                               data_root=data_root,\
                               data_mode1 = args.data_mode1,\
                               data_mode2 = args.data_mode2,\
                               inter_num = args.inter_num,\
                               intra_num = args.intra_num,temp=args.temporal)  
                               
    else:                                
        raise Exception("Not supported dataset!!!!") 
        
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__() # total number of steps per epoch
    val_data = DataLoader(dataset_val, batch_size=args.batch_size_val, pin_memory=True, num_workers=8)
        
    if torch.device("cuda") == device:
        rank = args.local_rank if args.local_rank >=0 else 0
        device_info = "CUDA Device {} is: {}".format(rank, torch.cuda.get_device_name(args.local_rank))
        log_writer.write(device_info, log_all=True)
    
    log_writer.write("Starting training...")
    log_writer.write("Each epoch includes {} iterations".format(args.step_per_epoch))
    train_timer = Timer()
    snapshot_timer = Timer()
    max_step = args.step_per_epoch*args.epoch
    if args.resume is True:
        step = checkpoint_info['best_monitored_iteration'] + 1
        start_epoch = checkpoint_info['best_monitored_epoch']
        best_dict = checkpoint_info
    else:    
        step = 0 # total training steps across all epochs
        start_epoch = 0
        best_dict={
            'best_monitored_value': 0,
            'best_psnr':0,
            'best_ssim':0,
            'best_monitored_iteration':-1,
            'best_monitored_epoch':-1, 
            'best_monitored_epoch_step':-1,
        }
    
    
    epoch_timer = Epoch_Timer('m')
    for epoch in range(start_epoch,args.epoch):
        sampler.set_epoch(epoch) ## to shuffle data
        if step > max_step:
            break
        epoch_timer.tic()
        for i, all_data in enumerate(train_data): 
            data = all_data[0]
            img_ids = all_data[1]           
            # data: 4d tensor,(batch_size,3*input_num+3*output_num,h,w) BGR format
            data_gpu = data.to(device, non_blocking=True) / 255. 
            imgs_tensor = data_gpu[:, :3*args.input_num] # (t2b,b2t)/(blur,RS)/(pre,cur) (batch_size,3*2,h,w)
            gts_tensor = data_gpu[:, 3*args.input_num:]  # multi gts   (batch_size,3*output_num,h,w)
            learning_rate = get_learning_rate(step)
            
            ##### Temporal-order encoding 
            batch,_,height,width = imgs_tensor.shape
            rs_encode = torch.arange(0,height).type_as(imgs_tensor).unsqueeze(1).repeat(1,width) ##(h,w)
            latent_gs_encode = []
            for out_i in range(0,args.output_num):
                gs_encode = torch.Tensor([(height-1)//(args.output_num-1)*out_i]).type_as(imgs_tensor).unsqueeze(0).repeat(height,width) #(h,w)
                latent_gs_encode.append(gs_encode)
            latent_gs_encodes = torch.stack(latent_gs_encode,dim=0)  ##(output_num,h,w)
            ### Relative location of i_th latent gs to input rs
            latent_gs_encodes = rs_encode.unsqueeze(0) - latent_gs_encodes  ##(output_num*1,h,w)
            latent_gs_encodes = latent_gs_encodes.unsqueeze(0).repeat(batch,1,1,1) ##(batch,output_num*1,h,w)
            
            
            pred, info = model.update(imgs_tensor,latent_gs_encodes, gts_tensor, learning_rate, training=True)     
            img_height = pred.shape[-2]
            MAX_DIFF = 1 
            mse = ((gts_tensor - pred) * (gts_tensor - pred)).reshape(args.batch_size*args.output_num,3,img_height,-1)
            mse = torch.mean(torch.mean(torch.mean(mse,-1),-1),-1).detach().cpu().data
            psnr_aa = 10* torch.log10( MAX_DIFF**2 / mse ) 
            psnr = torch.mean(psnr_aa)
            
            ssim = ssim_matlab(gts_tensor.contiguous().view(args.batch_size*args.output_num,3,img_height,-1)\
                              ,pred.contiguous().view(args.batch_size*args.output_num,3,img_height,-1)).detach().cpu().numpy()

            ##### Write summary to tensorboard
            if is_main_process():
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/charb', info['loss_charb'], step)
                writer.add_scalar('loss/total', info['loss_total'], step)
                writer.add_scalar('psnr', psnr, step)  
                writer.add_scalar('ssim', float(ssim), step)  
            
            ### Log traing info to screen and file
            should_print = (step % 1000 == 0 and step !=0)  # 1000
            extra = {}
            if should_print is True:
                extra.update(
                    {
                        "lr": "{:.2e}".format(learning_rate),
                        "time": train_timer.get_time_since_start(),
                        "train/total_loss":format(info['loss_total'].detach().cpu().numpy(), '.4f' ),
                        "train/loss_charb":format(info['loss_charb'].detach().cpu().numpy(),'.4f'),
                        "train/psnr":format(psnr,'.4f'),
                        "train/ssim":format(ssim,'.4f'),
                    }
                )
            
                train_timer.reset()
                val_infor = evaluate(model, val_data, step,writer_val,True)
                extra.update(val_infor)
            
            _summarize_report(
                                should_print=should_print,
                                extra=extra,
                                prefix=args.dataset_name+'_'+args.data_mode1+'-'+args.data_mode2+'_'+str(args.input_num)+'_'+str(args.output_num),
                                log_writer = log_writer,
                                current_iteration=step,
                                max_iterations=max_step
                                )
                                
            #### Conduct full evaluation and save checkpoint
            if step % 5000 == 0 and step !=0:    # 5000
                log_writer.write("Evaluation time. Running on full validation set...")
                all_val_infor = evaluate(model, val_data, step,writer_val,False,use_tqdm=True)
                val_extra = {"validation time":snapshot_timer.get_time_since_start()}
                if (all_val_infor['val/ssim']+all_val_infor['val/psnr'])/2 > best_dict['best_monitored_value']:
                    best_dict['best_monitored_iteration'] = step    
                    best_dict['best_monitored_epoch_step'] = i
                    best_dict['best_monitored_epoch'] = epoch
                    best_dict['best_monitored_value'] = float(format((all_val_infor['val/ssim']+all_val_infor['val/psnr'])/2,'.4f'))
                    best_dict['best_ssim'] = all_val_infor['val/ssim']
                    best_dict['best_psnr'] =all_val_infor['val/psnr']
                    model.save_model(args,step,best_dict, update_best=True) 
                else:
                    model.save_model(args,step,best_dict, update_best=False) 
                
                val_extra.update(
                    {'current_psnr':all_val_infor['val/psnr'],
                     'current_ssim':all_val_infor['val/ssim'],
                    }
                )
                val_extra.update(best_dict)
                prefix = "{}: full val".format(args.dataset_name+'_'+args.data_mode1+'-'+args.data_mode2+'_'+str(args.input_num)+'_'+str(args.output_num)) 
                _summarize_report(
                                extra=val_extra,
                                prefix=prefix,
                                log_writer = log_writer,
                                current_iteration=step,
                                max_iterations=max_step
                                )
                
                snapshot_timer.reset()
                gc.collect() # clear up memory
                if device == torch.device("cuda"):
                    torch.cuda.empty_cache()

            step += 1
            if step > max_step:
                break

        if is_main_process():
            print("EPOCH: %02d    Elapsed time: %4.2f " % (epoch+1, epoch_timer.toc()))
        dist.barrier()



def evaluate(model, val_data, step,writer_val,single_batch,use_tqdm=False):   

    psnr_list = []
    ssim_list = []
    disable_tqdm = not use_tqdm
    for i, all_data in enumerate(tqdm(val_data,disable=disable_tqdm)):
        data = all_data[0]
        img_ids = all_data[1]
        data_gpu = data.to(device, non_blocking=True) / 255.
        imgs_tensor = data_gpu[:, :3*args.input_num] 
        gts_tensor = data_gpu[:, 3*args.input_num:]
        
        ##### Temporal-order encoding 
        batch,_,height,width = imgs_tensor.shape
        rs_encode = torch.arange(0,height).type_as(imgs_tensor).unsqueeze(1).repeat(1,width) ##(h,w)
        latent_gs_encode = []
        for out_i in range(0,args.output_num):
            gs_encode = torch.Tensor([(height-1)//(args.output_num-1)*out_i]).type_as(imgs_tensor).unsqueeze(0).repeat(height,width) #(h,w)
            latent_gs_encode.append(gs_encode)
        latent_gs_encodes = torch.stack(latent_gs_encode,dim=0)  ##(output_num,h,w)
        ### relative location of ith latent gs to input rs
        latent_gs_encodes = rs_encode.unsqueeze(0) - latent_gs_encodes  ##(output_num*1,h,w)
        latent_gs_encodes = latent_gs_encodes.unsqueeze(0).repeat(batch,1,1,1) ##(batch,output_num*1,h,w)

        with torch.no_grad():
            pred, info = model.update(imgs_tensor,latent_gs_encodes, gts_tensor, training=False)   
        img_height = pred.shape[-2]
        
        for j in range(gts_tensor.shape[0]):

            MAX_DIFF = 1 
            mse = ((gts_tensor[j] - pred[j]) * (gts_tensor[j] - pred[j])).reshape(args.output_num,3,img_height,-1)
            mse = torch.mean(torch.mean(torch.mean(mse,-1),-1),-1).detach().cpu().data 
            psnr_aa = 10* torch.log10( MAX_DIFF**2 / mse )
            psnr = torch.mean(psnr_aa)
            psnr_list.append(psnr)
            ssim = ssim_matlab(gts_tensor[j].contiguous().view(args.output_num,3,img_height,-1),\
                               pred[j].contiguous().view(args.output_num,3,img_height,-1)).cpu().numpy()
            ssim_list.append(ssim)
        
        if single_batch is True:
            break
        
    if is_main_process() and single_batch is False:
       writer_val.add_scalar('psnr', np.array(psnr_list).mean(), step)
       writer_val.add_scalar('ssim', np.array(ssim_list).mean(), step)

    return {
            'val/ssim': float(format(np.mean(ssim_list),'.4f')),
            'val/psnr': float(format(np.mean(psnr_list),'.4f')),
            }



if __name__ == "__main__":    
    
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank) 
    
    # For reproduction 
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # To accelerate training process when network structure and inputsize are fixed
    torch.backends.cudnn.benchmark = True
    model = Model(config=args,\
                  local_rank=args.local_rank)
    train(model)
        
