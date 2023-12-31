# **********************************
# longzili logger for easier training loop

'''
A custom logger class designed to simplify log management in the training loop.
Note: This class should be initialized after torch.distributions.

Arguments:
    log_name: The name of the log.
    project_name: The name of the project.
    resume: Whether to resume training.
    config_opt: The argument object containing all the hyperparameters.
    log_root_path: The root path for the logs.
    checkpoint_root_path: The root path for the checkpoints.
    tensorboard_root_path: The root path for TensorBoard.
    use_wandb: Whether to use wandb for log management.
    wandb_root_path: The root path for wandb.
    log_interval: The interval for log printing.

Usage:
    # 1. Initialize the logger.
    self.logger = Longzili_Logger(
        log_name=str(wandb_dir),
        project_name=opt.wandb_project,
        config_opt=opt,
        checkpoint_root_path=opt.checkpoint_root,
        tensorboard_root_path=opt.tensorboard_root,
        wandb_root_path=opt.wandb_root,
        use_wandb=True,
        log_interval=opt.log_interval,)

    # 2. Use the methods inside.
        2.1 Use `tick` to record steps in the training loop.
        2.2 Use `log_info`, `log_scalar`, `log_image` to record logs.
        2.3 Use the following methods to log epoch:
            self.logger.log_scalar(force=True, log_type='epoch', 
                                    training_stage='train')
            self.logger.log_scalar(force=True, log_type='epoch', 
                                    training_stage='val')
        2.4 Use log_interval to control the output frequency
'''

# **********************************

import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import wandb
import logging
import os

import collections


def only_on_rank0(func):
    '''wrapper for only log on the first rank'''
    def wrapper(self, *args, **kwargs):
        if self.rank != 0:
            return
        return func(self, *args, **kwargs)
    return wrapper


class Longzili_Logger(object):
    def __init__(self, 
                log_name: str,
                project_name: str,
                resume = False,
                config_opt = None,
                log_root_path = None,
                checkpoint_root_path = None,
                tensorboard_root_path = None,
                use_wandb=False, 
                wandb_root_path = None,
                log_interval=100,
                log_interval_image = None):

        # check device and so on 
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.cards = torch.distributed.get_world_size()
        else:   
            self.rank = 0
            self.cards = 1  
        
        print(f'training on {self.cards} cards')
        # define root path
        if use_wandb and wandb_root_path is None:
            wandb_root_path = './logger'
        if log_root_path is not None:
            tensorboard_root_path = os.path.join(log_root_path, 'tensorboard')
            checkpoint_root_path = os.path.join(log_root_path, 'checkpoint')
        else:
            if tensorboard_root_path is None:
                tensorboard_root_path = './logger'
            if checkpoint_root_path is None:
                checkpoint_root_path = './logger'
        # define path
        self.use_wandb = use_wandb
        if use_wandb:
            self.wandb_path = os.path.join(wandb_root_path, log_name)
        self.tensorboard_path = os.path.join(tensorboard_root_path, log_name)
        self.checkpoint_path = os.path.join(checkpoint_root_path, log_name)
        self.create_directories(tensorboard_root_path, checkpoint_root_path)

        # initialize state
        self.log_interval = log_interval
        self.log_interval_image = log_interval * 30 if log_interval_image is None else log_interval_image
        self.step = 0
        self.values = {'train': {}, 'val': {}, 'test': {}}   

        # initialize mg_logger, tensorboard, wandb and so on
        if self.rank == 0:
            self.mg_logger = logging.getLogger(__name__)
            handler = logging.FileHandler(os.path.join(checkpoint_root_path, f"mg_logger.log"))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.mg_logger.addHandler(handler)
            self.mg_logger.setLevel(logging.INFO)
            # initialize tensorboard
            self.tb_logger = self._tensorboard_init(self.tensorboard_path, resume=resume)
            # initialize wandb
            if use_wandb:
                self.wandb_logger = self._wandb_init(log_name, project_name, self.wandb_path, config_opt, resume = resume)


    @only_on_rank0
    def log_image(self, tag, value, log_type="iter", force=False, step = 0, training_stage = 'train'):
        if log_type == "iter" and self.step % self.log_interval_image != 0 and not force:
            return
        elif log_type == "epoch" and not force:
            return
        value = value.detach().cpu()
        grid = torchvision.utils.make_grid(value, normalize=True)  # BCWH -> grid image
        grid = grid.permute(1, 2, 0)  # HWC
        grid = grid.numpy()
        tag = f'{training_stage}/{tag}'
        self.tb_logger.add_image(tag=tag, img_tensor=grid.transpose((2, 0, 1)), global_step = step, dataformats='CHW')
        # img = Image.fromarray((grid * 255).astype(np.uint8)) # convert to PIL Image
        if self.use_wandb:
            self.wandb_logger.log({tag: wandb.Image(grid)}, step=self.step)

    def log_scalar(self, tag=None, value=None, force=False, log_type="iter", log_list=None, training_stage='train', step = None):
        assert log_type in ["iter", "epoch"]
        assert training_stage in ["train", 'val', 'test', 'info']
        if step is not None:
            step = step
        else:
            step = self.step

        if tag is not None and value is not None:
            if tag not in self.values[training_stage]:
                self.values[training_stage][tag] = []
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            self.values[training_stage][tag].append(self._reduce_value(value))

        if not force:
            if log_type == "iter" and step % self.log_interval == 0:
                self._write_log_for_tags(log_list, log_type=log_type, training_stage=training_stage, step = step)
            elif log_type == "epoch":
                self._write_log_for_tags(log_list, log_type=log_type, training_stage=training_stage, step = step)
                self._clear_all_values(log_list, training_stage=training_stage) 
        else:
            self._write_log_for_tags(log_list, log_type=log_type, training_stage=training_stage, step = step)
            if log_type == "epoch":
                self._clear_all_values(log_list, training_stage=training_stage) 

    @only_on_rank0
    def log_info(self, tag, value, step=None, training_stage='info'):
        assert training_stage in ["train", 'val', 'test', 'info']
        if step is None:
            step = self.step
        self._write_log(tag, value, training_stage=training_stage, step=step)

    # @only_on_rank0
    def log_scalar_dict(self, dict_values = None, log_type="iter", force=False, training_stage='train'):
        assert isinstance(dict_values, dict)
        for tag, value in dict_values.items():
            self.log_scalar(tag, value, log_type=log_type, force=force, training_stage=training_stage)

    @only_on_rank0
    def log_image_dict(self, dict_images, log_type="iter", force=False, training_stage='train'):
        assert isinstance(dict_images, dict)
        for tag, value in dict_images.items():
            self.log_image(tag, value, log_type=log_type, force=force, training_stage = training_stage)

    @only_on_rank0
    def log_info_dict(self, dict_values, step=None, training_stage='info'):
        assert isinstance(dict_values, dict)
        if step is None:
            step = self.step
        for tag, value in dict_values.items():
            self._write_log(tag, value, step=step)

    @only_on_rank0
    def _write_log_for_tags(self, log_list, log_type='iter', training_stage='train', step = None):
        step = self.step if step is None else step
        for tag in self.values[training_stage]:
            if log_list is None or tag in log_list:  
                if not self.values[training_stage][tag]:  
                    continue
                if log_type == 'iter':
                    mean_value = np.mean(self.values[training_stage][tag][-self.log_interval:])  
                else:  
                    mean_value = np.mean(self.values[training_stage][tag])
                if training_stage in ['val', 'test']:
                    pass
                self._write_log(training_stage + '/' + log_type + '/' + tag, mean_value, step = step)

    @only_on_rank0
    def _write_log(self, tag, value, step):
        self.tb_logger.add_scalar(tag, value, global_step = step)
        if self.use_wandb:
            self.wandb_logger.log({tag: value}, step=step)

    def _clear_all_values(self, log_list, training_stage='train'):
        for tag in self.values[training_stage]:
            if log_list is None or tag in log_list:
                self.values[training_stage][tag] = []

    @only_on_rank0
    def _update_step(self):
        self.step += self.cards

    def tick(self):
        self._update_step()


    # ===== other base functions ====
    @staticmethod
    def _wandb_init(log_name, project_name, wandb_path, config_opt, resume = False):
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)
        wandb_logger = wandb.init(
                    project=project_name,
                    name=str(log_name),
                    dir=wandb_path,
                    resume = resume, 
                    config = config_opt, 
                    reinit = True,)
        return wandb_logger

    @staticmethod
    def _tensorboard_init(tensorboard_path, resume, flush_secs = 3):
        if resume:
            tb_logger = SummaryWriter(tensorboard_path, flush_secs=flush_secs, resume=True)
        else:
            tb_logger = SummaryWriter(tensorboard_path, flush_secs=flush_secs)
        return tb_logger

    def _reduce_value(self, value, average=True):
        try:
            world_size = torch.distributed.get_world_size()
        except Exception as err:
            world_size = 1
        if world_size < 2:  # single GPU
            return value
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        if not value.is_cuda:
            value = value.cuda(self.rank)
        with torch.no_grad():
            torch.distributed.all_reduce(value)   # get reduce value
            if average:
                value = value.float()
                value /= world_size
        return value.cpu().item()

    @staticmethod
    def create_directories(*paths):
        for path in paths:
            os.makedirs(path, exist_ok=True)




def get_grid_from_list(plot_list, n_rows=2, n_cols=0):
    assert isinstance(plot_list[0], torch.Tensor)

    n = len(plot_list)
    if n_cols == 0:
        n_cols = (n + n_rows - 1) // n_rows

    N, C, H, W = plot_list[0].shape
    grid = torch.zeros(N, C, H * n_rows, W * n_cols)

    for idx, tensor in enumerate(plot_list):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        grid[:, :, row_idx * H:(row_idx + 1) * H, col_idx * W:(col_idx + 1) * W] = tensor

    return grid

def get_grid_from_dict(plot_dict, n_rows=2, n_cols=0):
    plot_list = []
    for key, value in plot_dict.items():
        plot_list.append(value)
    return get_grid_from_list(plot_list, n_rows, n_cols)

def get_grid_from_dict_auto(plot_dict, n_rows=2, n_cols=0):
    assert isinstance(plot_dict, dict)

    shape_dict = collections.defaultdict(list)
    for key, value in plot_dict.items():
        if isinstance(value, torch.Tensor):
            shape_dict[value.shape[2:]].append((key, value))

    new_plot_dict = {}
    for shape, kv_list in shape_dict.items():
        n = len(kv_list)
        if n_cols == 0:
            n_cols = (n + n_rows - 1) // n_rows

        big_grid = torch.zeros((kv_list[0][1].shape[0], kv_list[0][1].shape[1], shape[0] * n_rows, shape[1] * n_cols))
        
        grid_key_info = []
        for idx, (key, tensor) in enumerate(kv_list):
            row_idx = idx // n_cols
            col_idx = idx % n_cols
            big_grid[:, :, row_idx * shape[0]:(row_idx + 1) * shape[0], col_idx * shape[1]:(col_idx + 1) * shape[1]] = tensor
            grid_key_info.append(f"{row_idx + 1}-{col_idx + 1}:{key}")
        
        new_key = ', '.join(grid_key_info)
        new_plot_dict[new_key] = big_grid

    return new_plot_dict
