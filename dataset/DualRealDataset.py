import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1) # Avoid deadlock when using DataLoader

class DualRealDataset(Dataset):
    def __init__(self, dataset_cls, input_num, output_num,data_root,data_mode1,data_mode2,inter_num,intra_num,temp=False):
        self.inter_num = inter_num
        self.intra_num = intra_num
        self.data_mode1 = data_mode1
        self.data_mode2 = data_mode2
        self.dataset_cls = dataset_cls #'train','test','validate'
        self.input_num = input_num
        self.output_num = output_num
        self.data_root = data_root 
        self.image_root = os.path.join(self.data_root, self.dataset_cls)
        self.temporal = temp
        
        self.prepare_data()
    
    def is_image(self, img):
        img_types = ['.PNG','.png','.JPG','.jpg','.JPEG','.jpeg']
        ext_name = os.path.splitext(img)[-1]
        if ext_name in img_types:
            return True
        else:
            return False  

    def get_seq(self, seq_name, data_mode):
        seq_path = os.path.join(self.image_root, seq_name)
        if data_mode == 'Blur':
            seq_mode_path = os.path.join(seq_path,'GS','RGB')
        elif data_mode == 'Sharp':
            seq_mode_path = os.path.join(seq_path,'HS','RGB')
        elif data_mode == 'RS':
            seq_mode_path = os.path.join(seq_path,'RS','RGB')
        elif data_mode == 'iRS':
            seq_mode_path = os.path.join(seq_path,'iRS','RGB')
        elif data_mode == 'RSGR':
            seq_mode_path = os.path.join(seq_path,'RSGR','RGB')
        else:
            raise Exception('Not suppoted data mode!!!')
        
        seq_mode = [img for img in os.listdir(seq_mode_path) if self.is_image(img)]
        ### !!!!!!!!!
        seq_mode = sorted(seq_mode) ####  seq_mode = sorted(seq_mode,key=lambda x:int(os.path.splitext(x)[0])) 
        
        if data_mode == 'Sharp':
            seq_mode = seq_mode[(self.intra_num//2):][::(self.inter_num+self.intra_num)]
        
        return [os.path.join(seq_mode_path,img) for img in seq_mode]
        
         
    def prepare_data(self):
        self.sample_paths = []
        seq_list = os.listdir(self.image_root) 
        seq_list = sorted(seq_list)
        
        for seq_name in seq_list:
            
            seq_mode1 = self.get_seq(seq_name, self.data_mode1)
            seq_mode2 = self.get_seq(seq_name, self.data_mode2)
            
            seq_gts_path = os.path.join(self.image_root,seq_name,'HS','RGB')
            seq_gts = [img for img in os.listdir(seq_gts_path) if self.is_image(img) ]
            ### !!!!!
            seq_gts = sorted(seq_gts)   #### seq_gts = sorted(seq_gts,key=lambda x:int(os.path.splitext(x)[0])) 
            seq_gts = [os.path.join(seq_gts_path,img) for img in seq_gts]
            
            assert len(seq_mode1) == len(seq_mode2)
            
            for idx in range(0,len(seq_mode1)):
            
                # !!!!!!!!
                if self.temporal and (idx+1)> len(seq_mode1)-1:
                    break
                
                t2b = seq_mode1[idx]
                gt = []
                
                ### !!!!!!!!
                if self.temporal:
                    b2t = seq_mode1[idx+1]
                    all_gts = seq_gts[(idx+1)*(self.inter_num+self.intra_num):(idx+1)*(self.inter_num+self.intra_num)+self.intra_num]
                else:
                    b2t = seq_mode2[idx]
                    all_gts = seq_gts[idx*(self.inter_num+self.intra_num):idx*(self.inter_num+self.intra_num)+self.intra_num]              
                
                
                assert len(all_gts) == self.intra_num
                if self.output_num == 1:
                    gt.append(all_gts[len(all_gts)//2])
                else:
                    for order in range(1,self.output_num+1):
                        k = round((order-1)/(self.output_num-1) * (len(all_gts)-1)) + 1
                        gt.append(all_gts[k-1] )
                
                target_dict = {
                 't2b':t2b,\
                 'b2t':b2t,\
                 'gt':gt
                } 
                self.sample_paths.append(target_dict) 

    def __len__(self):
        return len(self.sample_paths)

    """data augmentation- random corpping"""
    def aug(self, imgs_arr, gts_arr, h, w):
        ih, iw, _ = imgs_arr.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs_arr = imgs_arr[x:x+h, y:y+w, :]
        gts_arr = gts_arr[x:x+h, y:y+w, :]
        return imgs_arr, gts_arr

    def center_corp(self, img, delta_x,delta_y):
        # img (h,w,c)
        h,w,_ = img.shape
        new_w = w-2*delta_x
        new_h = h-2*delta_y
        return img[delta_y:delta_y+new_h,delta_x:delta_x+new_w,]
        
    def getimg(self, index):
        
        target_dict = self.sample_paths[index]
        gts_list = []
        imgs_list = []
        # Load images
        t2b = cv2.imread(target_dict['t2b']) # <class 'numpy.ndarray'>, (h,w,c)
        imgs_list.append(t2b)
        b2t = cv2.imread(target_dict['b2t'])
        imgs_list.append(b2t)
        imgs_arr = np.concatenate(imgs_list,2) ## (t2b,b2t),3-d ndarray,(h,w,3*input_num)
        for gt_path in target_dict['gt']:
            gts_list.append(cv2.imread(gt_path))
        gts_arr = np.concatenate(gts_list,2) ## multiple gts,3-d ndarray,(h,w,3*output_num)
        
        return imgs_arr, gts_arr
            
    def __getitem__(self, index):         
        imgs_arr, gts_arr = self.getimg(index)

        if self.temporal:
            cur_img_path = os.path.splitext(self.sample_paths[index]['b2t'])[0]
        else:
            cur_img_path = os.path.splitext(self.sample_paths[index]['t2b'])[0]
            
        cur_img_id = cur_img_path.replace(self.image_root,'')        
        gts_ids = [ os.path.splitext(path)[0].replace(self.image_root,'') for path in self.sample_paths[index]['gt']]

        if self.dataset_cls == 'train':
            imgs_arr, gts_arr = self.aug(imgs_arr, gts_arr, 512, 512)
            ## further augmentation 
            if random.uniform(0, 1) < 0.5: # horizontal flipping
                imgs_arr = imgs_arr[:, ::-1]
                gts_arr = gts_arr[:, ::-1]
            
            if random.uniform(0, 1) < 0.5:    ### reverse rgb channel 
                height, width = imgs_arr.shape[0],imgs_arr.shape[1]
                imgs_arr = imgs_arr.reshape(height,width,self.input_num,3)[:,:,:,::-1]
                imgs_arr = imgs_arr.reshape(height,width,self.input_num*3)
                gts_arr = gts_arr.reshape(height,width,self.output_num,3)[:,:,:,::-1]
                gts_arr = gts_arr.reshape(height,width,self.output_num*3)
            
        imgs_arr = torch.from_numpy(imgs_arr.copy()).permute(2, 0, 1) # change from (h,w,c) to (c,h,w)
        gts_arr = torch.from_numpy(gts_arr.copy()).permute(2, 0, 1)
        return torch.cat((imgs_arr, gts_arr), 0),cur_img_id,gts_ids # 3-d ndarray,(3*input_num+3*output_num,h,w)

