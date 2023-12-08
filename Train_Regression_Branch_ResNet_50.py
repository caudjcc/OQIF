import os
import re
import copy
import glob
import time
import tqdm
import h5py
import json
import random
import logging
import numpy as np
import datetime as dt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import transforms
from typing import List,Tuple,NoReturn,Optional,Union

def loadJson(file_name):
    '''简易加载json'''
    with open(file_name,'r') as f:
        js = json.load(f)
    return js

def saveJson(file, file_name):
    '''简易保存json'''
    with open(file_name,'w') as f:
        json.dump(file,f)
        
def setup_logger(log_file, show: bool = False):
    '''
    生成一个log对象，但是其句柄会一直叠加，需要手动删除
    @param show：是否在控制台显示
    '''
    logger = logging.getLogger('Train')
    formatter = logging.Formatter(LOG_FORMAT)
    # 配置文件句柄
    fhandler = logging.FileHandler(log_file)
    fhandler.setLevel(logging.DEBUG)
    fhandler.setFormatter(formatter)
    # 配置控制台句柄
    chandler = logging.StreamHandler()
    chandler.setLevel(logging.DEBUG)
    chandler.setFormatter(formatter)
    # 添加句柄
    logger.addHandler(fhandler)
    if show:
        logger.addHandler(chandler)
    logger.setLevel(logging.INFO)
    return logger
    
# 学习率调整策略
def step_lr(epoch):
    '''step_lr'''
    gamma = 0.95
    stepsize = 1
    current_epoch = epoch - warm_up_epochs
    return gamma**(current_epoch//stepsize)
    
def multistep_lr(epoch):
    '''multistep lr'''
    gamma = 0.1
    lr_milestones = [20,40]
    return gamma**len([m for m in lr_milestones if m <= epoch])

def cosine_lr(epoch):
    '''余弦学习率变换'''
    step = [10,20,40]
    current_epoch = epoch - warm_up_epochs
    idx_list = [current_epoch//s for s in step]
    idx = idx_list.index(min(idx_list))
    step_cosine = step[idx]
    return 0.5 * (np.cos(current_epoch * 2 * np.pi / step_cosine) + 1)

def warm_up(epoch):
    '''warm up'''
    if epoch < warm_up_epochs:
        return (epoch+1)/warm_up_epochs
    if epoch >= warm_up_epochs:
        return 1

def warm_up_with_multistep_lr(epoch):
    '''融合分段学习率调整策略'''
    if epoch < warm_up_epochs:
        return (epoch+1)/warm_up_epochs
    if epoch >= warm_up_epochs:
        return multistep_lr(epoch)
    
def warm_up_with_cosine_lr(epoch):
    '''融合余弦调整学习率策略'''
    if epoch < warm_up_epochs:
        return (epoch+1)/warm_up_epochs
    if epoch >= warm_up_epochs:
        return cosine_lr(epoch)
    
def warm_up_with_step_lr(epoch):
    '''融合固定步长调整学习率策略'''
    if epoch < warm_up_epochs:
        return (epoch+1)/warm_up_epochs
    if epoch >= warm_up_epochs:
        return step_lr(epoch)
    
class ToTensor():
    '''
    转为tensor，并且转移到显存中（如果使用显卡）
    20220610 转到显卡中无法使用多进程
    '''
    def __init__(self, device):
        self.device = device
        self.device = 'cpu'
    
    def __call__(self, image:np.ndarray, label:float):
        image = image.transpose(2,0,1)
        image = torch.from_numpy(image).float().to(self.device)
        label = torch.tensor(label).float().to(self.device)
        return image, label
    
class CropAndResize():
    '''
    中心裁剪+resize
    '''
    def __init__(self, size:int = 96):
        '''
        @param:p:水平翻转和竖直翻转概率
        '''
        self.size = size
    
    def __call__(self, image:torch.tensor, label:torch.tensor):
        '''
        @param:image:图像
        @param:label:标签
        '''
        image = transforms.F.center_crop(image, min(image.shape[-2:]))
        image = transforms.F.resize(image, self.size)
        return image, label
    
class RandomLight():
    '''随机调整亮度'''
    def __init__(self, p):
        self.p = p
    def __call__(self, image, label):
        rnd = torch.rand(1).item()
        if rnd <= self.p:
            image = image*(torch.rand(1).item() + 0.5)
        return image, label
    
class RandomFlip():
    '''
    随机翻转
    '''
    def __init__(self, p:float= 0.5):
        '''
        @param:p:水平翻转和竖直翻转概率
        '''
        self.p = p
    
    def __call__(self, image:torch.tensor, label:torch.tensor):
        '''
        @param:image:图像
        @param:label:标签
        '''
        rnd = torch.rand(2)
        if rnd[0] <= self.p:
            image = transforms.F.hflip(image)
        if rnd[1] <= self.p:
            image = transforms.F.vflip(image)
        return image, label
    
class RandomRotate():
    '''
    随机旋转
    '''
    def __init__(self, p:float= 0.5, degrees=[-15,15]):
        '''
        @param:p:进入旋转操作的概率
        @param:degrees:旋转范围
        '''
        self.p = p
        self.degrees = degrees
        
    def __call__(self, image:torch.tensor, label:torch.tensor):
        '''
        @param:image:要旋转的图像
        @param:label:标签
        '''
        rnd = torch.rand(1).item()
        if rnd <= self.p:
            # 确定旋转角度
            angle = np.random.randint(self.degrees[0], self.degrees[1])
            image = transforms.F.rotate(image,angle)
        return image, label.squeeze(0)
        
class MyDataset():
    '''
    用于为模型准备数据
    '''
    def __init__(self, anno, phase, keep_ratio, transforms= None):
        '''
        @param:fns:样本名称
        '''
        self.anno  = anno
        self.phase = phase
        self.keep_ratio = keep_ratio
        self.size = len(anno['annotations'])
        np.random.shuffle(self.anno['annotations'])
        self.anno['annotations'] = self.anno['annotations'][:int(keep_ratio*self.size)]
        self.size = len(anno['annotations'])
        self.transforms = transforms
        
    def __call__(self):
        print('使用__getitem__(idx)获取样本号')
    
    def __len__(self):
        '''
        返回样本长度
        '''
        return self.size
    
    
    def __getitem__(self, idx):
        '''
        定义了__getitem__魔法函数，该类就可以下标操作了：[]
        '''
        anno = self.anno['annotations'][idx]
        image_id = anno['image_id']
        c,r,w,h = anno['bbox']
        label = anno['numrical_label']
        image_name = [d['file_name'] for d in self.anno['images'] if d['id'] == image_id][0]
        
        with h5py.File(os.path.join(SAVEROOT, self.phase, image_name),'r') as f:
            image = f['image'][r:r+h,c:c+w,:].astype(np.float32)
            
        if self.transforms:
            for transform in self.transforms:
                image,label = transform(image,label)
        return image, label.unsqueeze(0)/100
        
def train_model(model, data_loaders, criterion, optimizer, scheduler, num_epochs= 10, save= False):
    '''
    model: 需要训练的网络
    criterion：评价标准
    optimizer：优化器
    scheduler：学习率衰减
    num_epochs：迭代次数
    '''
    loss_list = {'train':[], 'val':[]}
    # 保存最好的模型与精度
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    try:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs), '-*' * 10)
            Logger_global.info('Epoch {}/{}{}'.format(epoch + 1, num_epochs, '-*' * 10))
            # 每个epoch都需要进行一次训练与验证的过程
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                loss_epoch = 0.0
                for idx, (inputs, labels) in enumerate(data_loaders[phase]):
                    if phase == 'train':
                        print('\r','training on batch {}'.format(idx), end= '')

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # 每个batch都需要重新计算梯度，所以清空
                    optimizer.zero_grad()

                    # 只有在训练过程中需要反向传播，所以测试过程中可以set_grad_enabled(Flase)
                    with torch.set_grad_enabled(phase == 'train'):
                        pred = model(inputs)
                        loss = criterion(pred, labels)
                        loss_multi = torch.pow(pred-labels, 2).mean(axis=0)
                        if phase == 'train':
                            loss_multi.backward(loss_multi.clone().detach())
                            optimizer.step()

                    # loss.item() 返回一个value，乘以一个input.size，防止最后一组input数量不等
                    loss_batch = loss.item() * inputs.size(0)
                    Logger_global.info( '{}ing on epoch {}, batch {}, loss: {:.5f}'.format(phase, epoch+1, idx, loss.item()))
                    loss_epoch += loss_batch

                # 求整个epoch的loss
                loss_epoch = loss_epoch / data_loaders[phase].dataset.__len__()
                loss_list[phase].append(loss_epoch)

                # 用验证集验证，记录最佳权值
                if phase == 'val':
                    print('\n', 'Val Loss: {:.5f}'.format(loss_list[phase][-1]))
                    Logger_global.info('Val Loss: {:.5f}'.format(loss_list[phase][-1]))
                    if loss_list[phase][-1] < best_loss:
                        best_loss = loss_list[phase][-1]
                        best_model_weights = copy.deepcopy(model.state_dict())
        
            # 学习率衰减
            scheduler.step()
    except KeyboardInterrupt:
        import traceback
        Logger_global.info(traceback.format_exc())
        print(traceback.format_exc())
    # 保存模型
    model.load_state_dict(best_model_weights)
    if save and epoch > 0:
        if not os.path.exists(save):
            os.makedirs(save)
        path_model = os.path.join(save, '{}_Loss_{:.6f}.pt'.format(NOWTIME, best_loss))
        torch.save(model.state_dict(), path_model)
        print('saved! {}'.format(path_model))
        Logger_global.info('saved! {}'.format(path_model))
    print('Best val Loss: {:.6f}'.format(best_loss))
    Logger_global.info('Best val Loss: {:.6f}'.format(best_loss))
    return model, loss_list

# *******log*******
LOG_FILE = 'log/{}_50.log'
LOG_FORMAT = '%(asctime)s-%(levelname)s-%(name)s-line:%(lineno)4d: %(message)s'
# *******anno*******
path_train = '../data/anno/coco_512/train.json' 
path_valid = '../data/anno/coco_512/valid.json'
path_test = '../data/anno/coco_512/test.json'
anno_train = loadJson(path_train)
anno_valid = loadJson(path_valid)
anno_test = loadJson(path_test)
SAVEROOT = '../data/anno/coco_512'
# *******warm_up*****
warm_up_epochs = 5
# *******batchsize*******
BATCHSIZE = 256

# data 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# 选择gpu/cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_transforms = [ToTensor(device), CropAndResize(96), RandomLight(0.5), RandomFlip(0.5), RandomRotate(0.5,[-30,30])]
valid_transforms = [ToTensor(device), CropAndResize(96), RandomLight(0.5)]
test_transforms  = [ToTensor(device), CropAndResize(96)]

train_set = MyDataset(anno_train, 'train', 1, train_transforms)
valid_set = MyDataset(anno_valid, 'valid', 1, valid_transforms)
test_set  = MyDataset(anno_test , 'test',  1, test_transforms)

train_loader = DataLoader(train_set,
                          batch_size=BATCHSIZE,
                          shuffle=True,
                          num_workers=6,
                          pin_memory=True,
                          drop_last=True)

valid_loader = DataLoader(valid_set,
                          batch_size=BATCHSIZE,
                          shuffle=False,
                          num_workers=6,
                          pin_memory=True,
                          drop_last=True)
data_loaders = {'train':train_loader, 'val':valid_loader}

# train
#seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
#log
if 'Logger_global' in dir():
    Logger_global.handlers = []
NOWTIME = dt.datetime.now().strftime('%Y%m%d%H')
log_file = LOG_FILE.format(NOWTIME)
Logger_global = setup_logger(log_file, show=False)
# 选择gpu/cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('training on {}'.format(device))
Logger_global.info('training on {}'.format(device))
# train
network = models.resnet50(weights=False)
network.conv1 = nn.Conv2d(125, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
network.fc = nn.Sequential(nn.Linear(2048,1),nn.Sigmoid())
network.to(device)
optimizer = optim.Adam(network.parameters(), lr=0.0001)
# LOSS
criterion = nn.MSELoss()
# 每步衰减至 gamma * 上个epoch学习率
Reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor = 0.5)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma= 0.95)
warm_up_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warm_up_with_step_lr)#参数同上
# 训练网络
path_save = '../models_reg/models_50_{}/'.format(NOWTIME[4:8])
num_epochs = 10
model, loss_list = train_model(network,
# model, loss_list = train_model(model_load,
                               data_loaders, 
                               criterion, 
                               optimizer, 
                               warm_up_lr_scheduler, 
                               num_epochs= num_epochs,
                               save= path_save)