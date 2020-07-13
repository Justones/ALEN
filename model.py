import torch
import torch.nn as nn
import torch.nn.functional as F

def space_to_depth(x):
    h,w = x.shape[2:4]
    output = torch.cat((x[:,:,0:h:2,0:w:2],
                        x[:,:,0:h:2,1:w:2],
                        x[:,:,1:h:2,1:w:2],
                        x[:,:,1:h:2,0:w:2]),dim=1)
    return output
def upsample(x):
    b,c,h,w = x.shape[0:4]
    avg = nn.AvgPool2d([4,4],stride=4)
    output = avg(x).view(b,c,-1)
    return output

class nonlocalblock(nn.Module):
    def __init__(self,channel=32,avg_kernel=2):
        super(nonlocalblock,self).__init__()
        self.channel = channel//4
        self.theta = nn.Conv2d(channel,self.channel,1)
        self.phi = nn.Conv2d(channel,self.channel,1)
        self.g = nn.Conv2d(channel,self.channel,1)
        self.conv = nn.Conv2d(self.channel,channel,1)
        self.avg = nn.AvgPool2d([avg_kernel,avg_kernel],stride=avg_kernel)
    def forward(self,x):
        H,W = x.shape[2:4]
        #u = F.interpolate(x,scale_factor=0.5)
        #avg = nn.AvgPool2d([2,2],stride=2)
        u=self.avg(x)
        b,c,h,w = u.shape[0:4]
        #avg = nn.AvgPool2d(5,stride=1,padding=2)
        #temp_x = torch.cat((x,avg(x)),dim=1)
        #avg = nn.AvgPool2d(11,stride=1,padding=5)
        #temp_x = torch.cat((temp_x,avg(x)),dim=1)
        theta_x = self.theta(u).view(b,self.channel,-1).permute(0,2,1)
        phi_x = self.phi(u)
        phi_x = upsample(phi_x)
        g_x = self.g(u)
        g_x = upsample(g_x).permute(0,2,1)

        #.view(batch_size,self.channel,-1)
        #theta_x = theta_x.permute(0,2,1)
        theta_x = torch.matmul(theta_x,phi_x)
        theta_x = F.softmax(theta_x,dim=-1)

        y = torch.matmul(theta_x,g_x)
        y = y.permute(0,2,1).contiguous()
        y = y.view(b,self.channel,h,w)
        y = self.conv(y)
        y = F.interpolate(y,size=[H,W])
        return y
        
class seblock(nn.Module):
    def __init__(self,in_channel=32,out_channel=32):
        super(seblock,self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_channel,out_channel),nn.ReLU(),nn.Linear(out_channel,out_channel),nn.Sigmoid())
    def forward(self,x):
        c,h,w = x.shape[1:4]
        avg = nn.AvgPool2d([h,w],stride=0)
        y = avg(x)
        y = y.view(1,1,1,-1)
        y = self.fc(y)
        y = y.view(1,-1,1,1)
        output = x*y
        return output
class Globalblock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Globalblock,self).__init__()
        self.fc = nn.Linear(in_channel,out_channel)
    def forward(self,x):
        c,w,h = x.shape[1:4]
        avg = nn.AvgPool2d([w,h],stride=0)
        x = avg(x)
        x = x.view(1,1,1,-1)
        x = self.fc(x)
        return x
class fusionblock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(fusionblock,self).__init__()
        self.fc = nn.Linear(in_channel,out_channel)
        self.fusion = nn.Sequential(nn.Conv2d(out_channel*2,out_channel,1),nn.LeakyReLU(0.2))
    def forward(self,x,y):
        w,h = y.shape[2:4]
        x = self.fc(x)
        x = x.view(1,-1,1,1)
        x = x.repeat(1,1,w,h)
        x = torch.cat((x,y),dim=1)
        x = self.fusion(x)
        return x
class single_block1(nn.Module):
    def __init__(self,in_channel=32,out_channel=32):
        super(single_block1,self).__init__()
        self.nonlocalblock = nonlocalblock(in_channel)
        self.seblock = seblock(2*in_channel,2*in_channel)
        self.fusion = nn.Sequential(nn.Conv2d(2*in_channel,out_channel,3,padding=1),nn.LeakyReLU(0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1),nn.LeakyReLU(0.2))
        #self.conv2 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1,dilation=1),nn.LeakyReLU(0.2))
    def forward(self,x):
        nonlocal_x = self.nonlocalblock(x)
        #nonlocal_x = self.nonlocalblock(seblock_x)
        x_cat = torch.cat((nonlocal_x,x),dim=1)
        #x_cat = x
        seblock_x = self.seblock(x_cat)
        #x_cat = torch.cat((seblock_x,x),dim=1)
        fusion_x = self.fusion(seblock_x)
        #fusion_x = self.fusion(x_cat)
        conv1 = self.conv1(fusion_x)
        #output = self.conv2(conv1)
        #return output
        return conv1
class single_block(nn.Module):
    def __init__(self,in_channel=32,out_channel=32):
        super(single_block,self).__init__()
        self.conv1 = nn.Sequential(seblock(in_channel,in_channel),nn.Conv2d(in_channel,out_channel,3,padding=1),nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1),nn.LeakyReLU(0.2))
    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2
class EnhanceNet(nn.Module):
    def __init__(self,channel=64):
        super(EnhanceNet,self).__init__()
        self.inc = single_block(16,32)#512
        '''
        self.layer1 = nn.Sequential(nn.Conv2d(32*4,32,1),single_block1(32,64))#256
        self.layer2 = nn.Sequential(nn.Conv2d(64*4,64,1),single_block1(64,128))#128
        self.layer3 = nn.Sequential(nn.Conv2d(128*4,128,1),single_block1(128,256))#64
        '''
        self.layer1 = nn.Sequential(nn.MaxPool2d(2),single_block1(32,64))#256
        self.layer2 = nn.Sequential(nn.MaxPool2d(2),single_block1(64,128))#128
        self.layer3 = nn.Sequential(nn.MaxPool2d(2),single_block1(128,256))#64
        #self.inter = nn.Sequential(nn.Conv2d(256*4,256,1),single_block(256,512))
        #self.up0 = nn.ConvTranspose2d(512,256,2,2)
        #self.inter_layer = nn.Sequential(single_block(512,256))

        #self.global_feature = Globalblock(256,256)
        #self.fusionblock0 = fusionblock(256,32)
        #self.fusionblock1 = fusionblock(256,64)
        #self.fusionblock2 = fusionblock(256,128)
        #self.fusionblock3 = fusionblock(256,256)

        self.up1 = nn.ConvTranspose2d(256,128,2,2)
        self.layer4 = nn.Sequential(single_block(256,128))
        self.up2 = nn.ConvTranspose2d(128,64,2,2)
        self.layer5 = nn.Sequential(single_block(128,64))
        self.up3 = nn.ConvTranspose2d(64,32,2,2)
        self.layer6 = nn.Sequential(single_block(64,32))
        self.output = nn.Sequential(nn.Conv2d(32,12,1),nn.ReLU())
    def forward(self,x):
        #x = space_to_depth(x)
        I = torch.cat((0.8*x,x,1.2*x,1.5*x),dim=1)
        inc = self.inc(I)
        '''
        layer1 = self.layer1(space_to_depth(inc))
        layer2 = self.layer2(space_to_depth(layer1))
        layer3 = self.layer3(space_to_depth(layer2))
        '''
        layer1 = self.layer1(inc)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        #global_feature = self.global_feature(layer3)
        #inc = self.fusionblock0(global_feature,inc)
        #layer1 = self.fusionblock1(global_feature,layer1)
        #layer2 = self.fusionblock2(global_feature,layer2)
        #layer3 = self.fusionblock3(global_feature,layer3)
        #inter = self.inter(space_to_depth(layer3))
        #up0 = self.up0(inter)
        #inter_layer = torch.cat((up0,layer3),dim=1)
        #inter_layer = self.inter_layer(inter_layer)
        up1 = self.up1(layer3)
        layer4 = torch.cat((up1,layer2),dim=1)
        layer4 = self.layer4(layer4)
        up2 = self.up2(layer4)
        layer5 = torch.cat((up2,layer1),dim=1)
        layer5 = self.layer5(layer5)
        up3 = self.up3(layer5)
        layer6 = torch.cat((up3,inc),dim=1)
        layer6 = self.layer6(layer6)
        output = self.output(layer6)
        output = F.pixel_shuffle(output,2)
        return output




