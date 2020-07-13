import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import argparse
from model import EnhanceNet
from dataset import myDataset
from torch.utils.data import DataLoader
from utils import save_images
from glob import glob
import torchvision.transforms as TF
#import torchvision.models as models
import torch.nn.functional as F
from ssimloss import MS_SSIM
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="3", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=1, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=60000, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=512, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=20, help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test/low', help='directory for testing inputs')
parser.add_argument('--train_dir', dest='train_dir', default='/data/zc/nonlocal/test_data/', help='directory for testing inputs')
parser.add_argument('--decom', dest='decom', default=0, help='decom flag, 0 for enhanced results only and 1 for decomposition results')
parser.add_argument('--continue_train',dest='continue_train',default=0)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
enhancenet = EnhanceNet()

print('#parameters: ',sum(param.numel() for param in enhancenet.parameters()))
#enhancenet = torch.nn.DataParallel(enhancenet,device_ids=[0,1,2])
loss_list = []
eps=1e-12
if args.use_gpu:
    enhancenet = enhancenet.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True
enhance_optim = torch.optim.Adam(enhancenet.parameters(),lr = args.start_lr)
w_log = SummaryWriter(log_dir='logs')
def tv_loss(S,I):
    S_gray = 0.299*S[:,0,:,:]+0.587*S[:,1,:,:]+0.114*S[:,2,:,:]
    S_gray = S_gray.unsqueeze(1)
    L = torch.log(S_gray+0.0001)
    #L = S_gray
    dx = L[:,:,:-1,:-1]-L[:,:,:-1,1:]
    dy = L[:,:,:-1,:-1]-L[:,:,1:,:-1]
    #print(dx,dy)
    p_alpha = 1.2
    p_lambda = 1.5
    dx = p_lambda/(torch.pow(torch.abs(dx),p_alpha)+0.00001)
    dy = p_lambda/(torch.pow(torch.abs(dy),p_alpha)+0.00001)
    #print(dx,dy)
    x_loss = dx*((I[:, :, :-1, :-1]-I[:, :, :-1, 1:])**2)
    y_loss = dy*((I[:, :, :-1, :-1]-I[:, :, 1:, :-1])**2)
    tvloss = 0.5*torch.abs(x_loss+y_loss).mean()
    return tvloss
ms_ssim_loss = MS_SSIM(max_val=1)
def ms_ssim(x,y):
    mu_x = torch.mean(x)
    mu_y = torch.mean(y)
    c1 = 1e-4
    c2 = 0.03
    l = (2*mu_x*mu_y+c1)/(mu_x**2+mu_y**2+c1)
    scale = 1
    for i in range(4):
        resize_x = F.interpolate(x,scale_factor=scale)
        resize_y = F.interpolate(y,scale_factor=scale)
        scale = scale*0.5
        sigma_x = torch.var(resize_x)
        sigma_y = torch.var(resize_y)
        #cov = torch.mean(resize_x*resize_y)-torch.mean(resize_x)*torch.mean(resize_y)
        cov = torch.mean((resize_x-torch.mean(resize_x))*(resize_y-torch.mean(resize_y)))
        s = (2*cov+c2)/(sigma_x+sigma_y+c2)
        l = l*s
    #print('losss: %f\n'%(float(l)))
    return 1.0-l
def final_loss(S_low,high_im):
    pix_loss = torch.abs(S_low-high_im).mean()
    ssim_loss = 1-ms_ssim_loss(S_low,high_im)
    beta = 0.85
    loss = beta*pix_loss+(1-beta)*ssim_loss
    return loss

def eval():
    eval_set = myDataset(args.train_dir,'eval',args.patch_size)
    with torch.no_grad():
        dataLoader = DataLoader(eval_set,batch_size=1)
        for (step,data) in enumerate(dataLoader):
            low_im,high_im = data
            low_im, high_im = low_im.cuda(), high_im.cuda()
            S_low = enhancenet(low_im)
            out = S_low
            mse = torch.abs(high_im-out).mean()
            print(float(mse))
            torch.cuda.empty_cache()
def train():
    f = open('loss.txt','w')
    f.close()
    start_epoch = 0
    print(args.continue_train)
    if args.continue_train:
        checkpoint = torch.load(args.ckpt_dir+'/state_final.pth')
        start_epoch = checkpoint['epoch']+1
        print(start_epoch)
        if start_epoch < args.epoch:
            print(start_epoch)
            enhancenet.load_state_dict(checkpoint['enhance'])
            enhance_optim.load_state_dict(checkpoint['enhance_optim'])
        else :
            pass
    print(start_epoch)
    train_set = myDataset(args.train_dir,'train',args.patch_size)
    sum_loss_f = 0.0
    for epoch in range(start_epoch,args.epoch):
        step_num = 0
        sum_loss = 0.0
        dataloader = DataLoader(train_set,batch_size=args.batch_size, shuffle=True,pin_memory=True)
        for (_, data) in enumerate(dataloader):
            low_im, high_im = data
            low_im, high_im = low_im.cuda(), high_im.cuda()
            enhance_optim.zero_grad()
            S_low = enhancenet(low_im)
            loss = final_loss(S_low,high_im)
            loss.backward()
            enhance_optim.step()
            sum_loss += float(loss)
            step_num += 1
        print('epoch: %d, loss: %f'%(epoch,sum_loss/step_num))
        sum_loss_f += sum_loss/step_num
        w_log.add_scalar('loss',sum_loss/step_num,epoch)
        #if epoch%200 ==0:
        #    eval()
        if epoch == 2000:
            for pa in enhance_optim.param_groups:
                pa['lr'] = args.start_lr/10.0
        elif epoch == 3000:
            for pa in enhance_optim.param_groups:
                pa['lr'] = args.start_lr/50.0
        elif epoch == 4000:
            for pa in enhance_optim.param_groups:
                pa['lr'] = args.start_lr/100.0
        if epoch % 50 ==1:
            f = open('loss.txt','a')
            if epoch > 1:
                f.write(str(sum_loss_f/50)+'\n')
            sum_loss_f = 0.0
            f.close()
            state = {'enhance':enhancenet.state_dict(),'enhance_optim':enhance_optim.state_dict(),'epoch':epoch}
            if not os.path.isdir(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
            torch.save(state,args.ckpt_dir+'/state_final.pth')
    state = {'enhance':enhancenet.state_dict(),'enhance_optim':enhance_optim.state_dict(),'epoch':epoch}
    torch.save(state,args.ckpt_dir+'/state_final.pth')
    w_log.close()

def test():
    checkpoint = torch.load(args.ckpt_dir+'/state_final.pth')
    enhancenet.load_state_dict(checkpoint['enhance'])
    print('load weights successfully')
    args.batch_size=1
    test_set = myDataset(args.train_dir,'test',args.patch_size)
    print('number of test samples: %d'%(test_set.len))
    dataLoader = DataLoader(test_set,batch_size=args.batch_size)
    with torch.no_grad():
        for (step,data) in enumerate(dataLoader):
            low_im,high_im,idstr,ratio = data
            low_im = low_im.cuda()
            print(low_im.shape)
            S_low = enhancenet(low_im)
            out = S_low
            out_cpu = out.cpu()
            out_cpu = np.minimum(out_cpu,1.0)
            out_cpu = np.maximum(out_cpu,0.0)
            save_images(os.path.join(args.save_dir,'%d_00_%d_out.png'%(idstr,ratio)),out_cpu.detach().numpy())
            save_images(os.path.join(args.save_dir,'%d_00_%d_gt.png'%(idstr,ratio)),high_im.detach().numpy())
            torch.cuda.empty_cache()
def main():  
    if args.use_gpu:
        if args.phase == 'train':
            train()
            test()
        elif args.phase == 'test':
            test()
        else:
            print('unknown phase')
    else:
        print('please use gpu')
if __name__ == '__main__':
    main()

