from torch.utils.data import Dataset
from glob import glob
import random
import torch
import os
import rawpy
import numpy as np
import torchvision.transforms as TF

def get_len(route,phase):
    if phase =='train':
        train_low_data_names = glob(route + 'short/0*_00*.ARW')
        train_low_data_names.sort()
        train_high_data_names = glob(route + 'long/0*_00*.ARW')
        train_high_data_names.sort()
        return len(train_high_data_names),train_low_data_names,train_high_data_names
    elif phase =='test':
        test_low_data_names = glob(route + 'short/1*_00*.ARW')
        #test_low_data_names = glob(route + 'short/*.ARW')
        test_low_data_names.sort()
        test_high_data_names = glob(route + 'long/1*_00*.ARW')
        #test_high_data_names = glob(route + 'long/*.ARW')
        test_high_data_names.sort()
        return len(test_low_data_names), test_low_data_names,test_high_data_names
    elif phase == 'eval':
        eval_low_data_names = glob(route + 'short/2*.ARW')
        eval_low_data_names.sort()
        eval_high_data_names = glob(route+'long/2*.ARW')
        eval_high_data_names.sort()
        return len(eval_high_data_names[0:1]), eval_low_data_names[0:1],eval_high_data_names[0:1]
    else:
        return 0, []
        
def pack_raw(raw):
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    #im = np.maximum(im-64,0)/(16383-64)
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    '''
    im1 = im[0:H:2,0:W:2,:]
    im2 = im[0:H:2,1:W:2,:]
    im3 = im[1:H:2,1:W:2,:]
    im4 = im[1:H:2,0:W:2,:]
    avg1 = im1.mean()
    avg2 = im2.mean()
    avg3 = im3.mean()
    avg4 = im4.mean()
    avg = (avg1+avg2+avg3+avg4)/4
    r1 = avg/avg1
    r2 = avg/avg2
    r3 = avg/avg3
    r4 = avg/avg4
    '''
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    
    
    
    '''
    temp = []
    for x in range(4):
        for y in range(4):
            temp.append(im[x:H:4,y:W:4,:])
    output = temp[0]
    for idx in range(15):
        output = np.concatenate((output,temp[idx+1]),axis=2)
    return output'''
    return out
class myDataset(Dataset):
    def __init__(self,route='./data/train/',phase='train',patch_size=48):
        self.route = route
        self.phase = phase
        self.patch_size = patch_size
        self.input_images = [None]*500
        self.gt_images = [None]*250
        self.num = [0]*250
        self.pre = [0]*250
        self.len, self.low_names,self.high_names = get_len(route,phase)
        print(len(self.low_names))
        if self.phase == 'train':
            j = 0
            for ids in range(self.len):
                gt_raw = rawpy.imread(self.high_names[ids])
                im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                self.gt_images[ids] = np.float32(im / 65535.0)
                idstr = os.path.basename(self.high_names[ids])[0:5]
                gt_exposure = float(os.path.basename(self.high_names[ids])[9:-5])
                self.pre[ids]=j
                while j < len(self.low_names):
                    if idstr == os.path.basename(self.low_names[j])[0:5] :
                        in_exposure = float(os.path.basename(self.low_names[j])[9:-5])
                        ratio = min(gt_exposure/in_exposure,300)
                        raw = rawpy.imread(self.low_names[j])
                        self.input_images[j] = pack_raw(raw)*ratio
                        self.num[ids] += 1
                        j += 1
                    else :
                        break

    def __getitem__(self,index):
        if self.phase == 'train':
            randomx = random.randint(0,self.num[index]-1)
            train_low_data = self.input_images[self.pre[index]+randomx]
            train_high_data = self.gt_images[index]
            h,w,_ = train_low_data.shape
            x = random.randint(0,h-self.patch_size)
            y = random.randint(0,w-self.patch_size)
            #mode = random.randint(0,7)
            #low_im = data_augmentation(train_low_data[x:x+self.patch_size,y:y+self.patch_size,:],mode)
            #high_im = data_augmentation(train_high_data[2*x:2*x+self.patch_size*2,2*y:2*y+self.patch_size*2,:],mode)
            low_im = train_low_data[x:x+self.patch_size,y:y+self.patch_size,:]
            high_im = train_high_data[2*x:2*x+self.patch_size*2,2*y:2*y+self.patch_size*2,:]
            if np.random.randint(2, size=1)[0] == 1:  # random flip
                low_im = np.flip(low_im, axis=0)
                high_im = np.flip(high_im, axis=0)
            if np.random.randint(2, size=1)[0] == 1:
                low_im = np.flip(low_im, axis=1)
                high_im = np.flip(high_im, axis=1)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                low_im = np.transpose(low_im, (1, 0, 2))
                high_im = np.transpose(high_im, (1, 0, 2))
            return TF.ToTensor()(low_im.copy()), TF.ToTensor()(high_im.copy())
        elif self.phase == 'test':
            #low_im = load_images(self.low_names[index])
            low_im = pack_raw(rawpy.imread(self.low_names[index]))
            idstr = os.path.basename(self.low_names[index])
            for ids in range(len(self.high_names)):
                tepstr = os.path.basename(self.high_names[ids])
                if idstr[0:5] == tepstr[0:5]:
                    in_exposure = float(idstr[9:-5])
                    gt_exposure = float(tepstr[9:-5])
                    ratio = min(gt_exposure/in_exposure,300)
                    gt_raw = rawpy.imread(self.high_names[ids])
                    im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                    high_im = np.float32(im / 65535.0)
                    return TF.ToTensor()(low_im*ratio),TF.ToTensor()(high_im),int(idstr[0:5]),ratio
                    #return TF.ToTensor()(low_im*ratio),TF.ToTensor()(high_im),int(a),ratio
        elif self.phase == 'eval':
            low_im = pack_raw(rawpy.imread(self.low_names[index]))
            in_exposure = float(os.path.basename(self.low_names[index])[9:-5])
            gt_raw = rawpy.imread(self.high_names[index])
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            high_im = np.float32(im / 65535.0)
            gt_exposure = float(os.path.basename(self.high_names[index])[9:-5])
            ratio = min(gt_exposure/in_exposure,300)
            return TF.ToTensor()(low_im*ratio),TF.ToTensor()(high_im)
    def __len__(self):
        return self.len
