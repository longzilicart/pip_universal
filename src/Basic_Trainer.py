
import numpy as np
# from sklearn.cluster import FeatureAgglomeration
import tqdm
import os
import itertools

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid as make_grid

# amp not support real-imag tensor, so not supported lama in torch1.7
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

import wandb
# import wandb

import sys
sys.path.append("..")
from longzili_utils.longzili_logger import *
from longzili_utils.longzili_loss_scaler import *
from longzili_utils.longzili_loadparam import *

# from utils.dataset_utils import *
from utils.image_io import *
from utils.imresize import *
from utils.loss_utils import *
from utils.schedulers import *
from utils.val_utils import *


def myprint(message, local_rank=0):
    # print on localrank 0 if ddp
    if dist.is_initialized():
        if dist.get_rank() == local_rank:
            print(message)
    else:
        print(message)


class Restoration_Trainer_Basic:
    '''Universal Restoration Trainer Basic
    Model:
    log:
        logger/tensorboard/wandb
    Train:
        fit function(fit, train, val wait for implement)
    '''
    def __init__(self,):
        
        # basic value
        self.iter = 0
        self.epoch = 0
        self.loss_scaler = LossScaler(max_len=100, scale_factor=5)
        self.min_loss_scaler = LossScaler(max_len=100, scale_factor=3)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__                               
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)    

    def init_adam_optimizer(self, net):
        self.optimizer = torch.optim.Adam(net.parameters(), lr = self.opt.lr, betas = (self.opt.beta1, self.opt.beta2))
        self.step_optimizer = LinearWarmupCosineAnnealingLR(optimizer=self.optimizer,warmup_epochs=10, max_epochs=100, warmup_start_lr=self.opt.lr/10)

    def init_adamw_optimizer(self, net):
        # initialize adamw 
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2), weight_decay=self.opt.weight_decay,)
        self.step_optimizer = LinearWarmupCosineAnnealingLR(optimizer=self.optimizer,warmup_epochs=10, max_epochs=150, warmup_start_lr=self.opt.lr/10)

    def init_sgd_optimizer(self, net):
        # initialize SGD optimizer
        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.opt.lr, momentum=0.9)
        self.step_optimizer = StepLR(self.optimizer, step_size = self.opt.step_size, gamma=self.opt.step_gamma)

    # ---- save load checkpoint ----
    @staticmethod
    def save_checkpoint(param, path, name:str, epoch:int):
        # simply save the checkpoint by epoch
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, name + '_{}_epoch.pkl'.format(epoch))
        torch.save(param, checkpoint_path)

    def save_model(self):
        '''basic save model'''
        net_param = self.net.module.state_dict() # default multicard
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        net_check = {
            'params': net_param,
            'epoch': self.epoch,
        }
        self.save_checkpoint(net_check, checkpoint_path, self.opt.checkpoint_dir + '-net', self.epoch)

    def save_opt(self):
        '''basic save opt'''
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        opt_param = self.optimizer.state_dict()
        step_opt_param = self.step_optimizer.state_dict() # if step opt is True
        opt_check = {
            'optimizer': opt_param,
            'step_optimizer': step_opt_param,
            'epoch' : self.epoch,
        }
        self.save_checkpoint(opt_check, checkpoint_path, self.opt.checkpoint_dir +'-opt', self.epoch)

    def load_model(self, strict=False):
        '''basic load model'''
        net_checkpath = self.opt.net_checkpath
        try:
            # net_checkpoint = torch.load(net_checkpath, map_location = 'cpu')
            net_checkpoint = torch.load(net_checkpath, map_location = 'cpu')['params']
        except Exception as err:
            net_checkpoint = torch.load(net_checkpath, map_location = 'cpu')
        # net_checkpoint = torch.load(net_checkpath, map_location = 'cpu')['params']
        
        if strict:
            self.net.load_state_dict(net_checkpoint['params'], strict=strict) # strict = False
        else:
            self.load_partial_state_dict(net_checkpoint)
        # epoch not need
        myprint('finish loading network')

    # @staticmethod
    def load_partial_state_dict(self, state_dict):
        """skip weight if the shape not match"""
        model = self.net
        own_state = model.state_dict()
        # myprint(own_state, state_dict)
        for name, param in state_dict.items():
            if name not in own_state:
                myprint(f'skip key not in the model: {name}')
                continue
            own_param = own_state[name]
            
            # the shape is match
            if own_param.shape == param.shape:
                own_param.copy_(param)
                
            # partial match
            elif len(own_param.shape) == len(param.shape):
                myprint(f"skip weight: {name}")
                continue
                min_channels = min(own_param.shape[1], param.shape[1])
                own_param.data[:, :min_channels].copy_(param.data[:, :min_channels])
                myprint(f"partial loaded: {name}")
            # do not match
            else:
                myprint(f"do not match, skip key: {name}")

        model.load_state_dict(own_state, strict=False)
        # return model

    def load_opt(self):
        '''basic load opt'''
        opt_checkpath = self.opt.opt_checkpath
        opt_checkpoint = torch.load(opt_checkpath, map_location = 'cpu')
        self.optimizer.load_state_dict(opt_checkpoint['optimizer'])
        self.step_optimizer.load_state_dict(opt_checkpoint['step_optimizer'])
        self.epoch = opt_checkpoint['epoch']
        myprint('finish loading opt')

    def resume(self):
        '''resume training
        '''
        if self.opt.net_checkpath is not None:
            self.load_model()
            myprint('finish loading model')
        else:
            raise ValueError('opt.net_checkpath not provided')

    def resume_opt(self,):
        if self.opt.resume_opt is True and self.opt.opt_checkpath is not None:
            self.load_opt()
            myprint('finish loading optimizer')
        else:
            myprint('opt.opt_checkpath not provided')     

    @staticmethod
    def load_state_dict_byhand(net, checkpoint, net_depth = [1,2], checkpoint_depth = [2,3]):
        def search_param_by_name(name, checkpoint_keys, checkpoint_num):
            for check_num in range(checkpoint_num, len(checkpoint_keys)):
                # for i in [3, 4]:
                for i in net_depth:
                    net_name = ".".join(name.split('.')[i:])
                    for j in checkpoint_depth:
                        check_name = ".".join(checkpoint_keys[check_num].split('.')[j:])
                        if net_name == check_name:
                            return check_num
            return -1
        checkpoint_keys = list(checkpoint.keys())
        checkpoint_num = 0
        params = net.named_parameters()
        for name, param in params:
            # myprint(name)
            num = search_param_by_name(name, checkpoint_keys, checkpoint_num)
            if num == -1:
                pass
            else:
                checkpoint_num = num
                myprint('loading {} from {}'.format(name, checkpoint_keys[checkpoint_num]))
                param.data = checkpoint[checkpoint_keys[checkpoint_num]] 
        myprint('finish loading')
    

    # ---- common loss function ----
    @staticmethod
    def pixel_loss(input, target, mode = 'l1'):
        assert mode in ['l1', 'sml1', 'l2']
        if mode == 'l1':
            L1loss = torch.nn.L1Loss(reduction = 'mean')
            loss = L1loss(input, target)        
        elif mode == 'sml1':
            smL1loss = torch.nn.SmoothL1Loss(reduction = 'mean')
            loss = smL1loss(input, target)
        elif mode == 'l2':
            mse_loss = torch.nn.MSELoss(reduction = 'mean')
            loss = mse_loss(input, target)
        else:
            raise ValueError('pixel_loss error: mode not in [l1,sml1,l2]')
        return loss


    def scale_pixel_loss(self, input, target, mode = 'l1'):
        assert mode in ['l1', 'sml1', 'l2']
        if mode == 'l1':
            L1loss = torch.nn.L1Loss(reduction = 'mean')
            loss = L1loss(input, target)        
        elif mode == 'sml1':
            smL1loss = torch.nn.SmoothL1Loss(reduction = 'mean')
            loss = smL1loss(input, target)
        elif mode == 'l2':
            mse_loss = torch.nn.MSELoss(reduction = 'mean')
            loss = mse_loss(input, target)
        else:
            raise ValueError('pixel_loss error: mode not in [l1,sml1,l2]')
        return self.loss_scaler(loss)


    @staticmethod
    def info_loss(input, target, mode = 'crossentropy'):
        assert mode in ['crossentropy', 'mine']
        if mode == 'crossentropy':
            infoloss = torch.nn.CrossEntropyLoss()
            loss = infoloss(input, target)
        elif mode == 'mine': 
            pass 
        return loss
    
    @staticmethod
    def prob_loss(input, target, mode = 'kl'):
        assert mode in ['kl']
        if mode == 'kl':
            loss = F.kl_div(input.softmax(dim=-1).log(), target.softmax(dim=-1), reduction='sum')
        return loss

    @staticmethod
    def mse_loss(input, target):
        return F.mse_loss(input, target)


    # reduce function
    def reduce_value(self, value, average=True):
        world_size = torch.distributed.get_world_size()
        if world_size < 2:  # single GPU
            return value
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        if not value.is_cuda:
            value = value.cuda(self.opt.local_rank)
        with torch.no_grad():
            dist.all_reduce(value)   # get reduce value
            if average:
                value = value.float()
                value /= world_size
        return value.cpu()

    def reduce_loss(self, loss, average=True):
        return self.reduce_value(loss, average=average)

    # ---- training fucntion ----
    def fit():
        raise ValueError('function fit() not implemented')

    def train():
        pass

    def val():
        pass
