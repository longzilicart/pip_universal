

import argparse
import os
import sys
import torch.multiprocessing as mp

from net.restormer import Restormer

from PIPNet_Trainer import *
from PIPNet_Tester import *
# pip network
from net.PIP_Net import PIPNet_Restormer_onskip_inter
# degradation aware module
from net.degradation_sensor import *
from net.PIP import *
# clip prompt



def get_parser():
    parser = argparse.ArgumentParser(description='Universal restoration')
    # logging interval by iter
    parser.add_argument('--log_interval', type=int, default=400, help='logging interval by iteration')
    # tensorboard
    parser.add_argument('--checkpoint_root', type=str, default='', help='where to save the checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='test', help='detail folder of checkpoint')
    parser.add_argument('--tensorboard_root', type=str, default='', help='root path of tensorboard, project path')
    parser.add_argument('--tensorboard_dir', type=str, required=True, help='detail folder of tensorboard')
    # wandb config
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='Sparse_CT')
    parser.add_argument('--wandb_root', type=str, default='')

    # DDP
    parser.add_argument('--local_rank', type=int, default = -1,
                        help = 'node rank for torch distributed training')
    # data_path
    parser.add_argument('--dataset_path', type=str, default = '', help='dataset root path for restoration task')
    
    
    # >>>>>>>>>> important dataset setting  >>>>>>>>>>>>>>>>
    parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                        help='which type of degradations is training and testing for.')

    parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
    # ========= dataset place do not modify =========
    parser.add_argument('--data_file_dir', type=str, default='data_dir/',      
                        action='store',
                        help='where clean images of denoising saves.')
    parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                        action='store',
                        help='where clean images of denoising saves.')
    parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                        action='store',
                        help='where training images of deraining saves.')
    parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                        action='store',
                        help='where training images of dehazing saves.')
    
    # ---- hyperparameter 
    # dataloader
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch_size')    
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='dataloader shuffle, False if test and val')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='dataloader num_workers, 4 is a good choice')
    parser.add_argument('--drop_last', default=True, type=bool,
                        help='dataloader droplast')
    # optimizer
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')  
    parser.add_argument('--weight_decay', default=0.01, type=float, 
                        help='weight decay of adamw, the default value in pytorch is 0.01')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='Adam beta1 args')    
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='Adam beta2 args')    
    parser.add_argument('--epochs', default=200, type=int,
                        help='training epochs')
    # step_optimizer
    parser.add_argument('--step_size', default=30, type=int,
                        help='StepLR step_size args')    
    parser.add_argument('--step_gamma', default=0.5, type=float,
                        help='StepLR step_gamma args')

    # checkpath && resume training
    parser.add_argument('--resume', default=False, type=bool,
                        help = 'resume network training or not, load network param')
    parser.add_argument('--resume_opt', default=False, type=bool,
                        help = 'resume optimizer or not, load opt param')
    parser.add_argument('--net_checkpath', default='', type=str,
                        help='network checkpath')
    parser.add_argument('--opt_checkpath', default='', type=str,
                        help='optimizer checkpath')
    
    # network hyper args
    parser.add_argument('--trainer_mode', default='train', type=str,
                        help = 'main function - trainer mode, train or test')
    parser.add_argument('--network', default='', type=str,
                        help='the network option, restormer, promptir, etc.')

    # option of loss function for prompt-in-prompt trainer 
    parser.add_argument('--high_reg_loss', default='cosine', type = str,
                        help = 'option of DDL loss, reg for high level prompt, [angle, cosine, None]')


    # # tester args
    parser.add_argument('--is_test', action='store_true', 
                    help='if --is_test, then is Trues')
    parser.add_argument('--is_save_image', action='store_true', 
                    help='if --is_save_image, then is True')
    parser.add_argument('--is_test_ood', action='store_true', 
                    help='if --is_test_ood, then is True, only on ood scnerios')
    parser.add_argument('--is_test_real', action='store_true', 
                    help='if --is_test_real, then is True, only on ood scnerios')
    parser.add_argument('--tester_save_path',default=None, type=str,
                        help='tester_save path, no use can be delete, add a default setting' )
    parser.add_argument('--tester_save_name', default='', type=str,
                        help='tester_save name' )

    return parser



def universal_main(opt):
    '''main function for universal trainer'''

    # 1. network selection
    if opt.network == 'promptir':
        # net = PromptIR(inp_channels=3, out_channels=3, decoder=True)  
        pass

    elif opt.network == 'PIPNet_Restormer_onskip_inter':
        decoder = True
        use_detask_label = True
        use_detask_prompt = True
        use_CLIP_prompt = False
        use_SAM_prompt = False
        use_degradation_sensor = False
        high_prompt_dim = 1
        low_prompt_dims = [64, 128, 256] 
        low_prompt_sizes = [64, 32, 16] # half shape
        degradation_num = 5 # degradation type to support
        net = PIPNet_Restormer_onskip_inter(inp_channels=3, out_channels=3,
                            decoder = decoder,
                            use_detask_label = use_detask_label,
                            use_detask_prompt = use_detask_prompt,
                            use_CLIP_prompt = use_CLIP_prompt,
                            use_SAM_prompt = use_SAM_prompt,
                            use_degradation_sensor = use_degradation_sensor,
                            high_prompt_dim = high_prompt_dim, 
                            low_prompt_dims = low_prompt_dims,
                            prompt_interaction_mode = 'pip_cross_topm',
                            degradation_num = degradation_num, 
                            low_prompt_sizes=low_prompt_sizes, 
                            )



    # degradation aware model, can be the sensor
    elif opt.network == 'VGG':
        # net = VGG('VGG16', class_num=5, withsoftmax=False) 
        net = ffc_resnet34(num_classes=5)
        # net = TV_resnet34(num_classes=5)
        # net = TV_vgg16(num_classes=5)
    else:
        raise ValueError(f'... Unsupported network: {opt.network}')


    # 2. get trainer 
    if opt.network in ['promptir']:
        # restore_trainer = PromptIR_Trainer(net, opt)    
        pass
    # pip trainer
    elif 'pip' in opt.network.lower():
        restore_trainer = PIPNet_Trainer(net, opt)
    # sensor 
    elif opt.network in ['VGG']:
        sensor_trainer = PIPNet_Sensor_Trainer(net, opt)

    # 3. train mode
    if opt.trainer_mode == 'train' and opt.network not in ['VGG']:
        restore_trainer.fit()
    elif opt.network in ['VGG']:
        sensor_trainer.fit()

    # 4. test mode
    elif opt.trainer_mode == 'test' or opt.is_test:
        tester = PIPNet_Tester(net, opt)
        tester.fit()
    
    else:
        raise ValueError('... option{trainer_mode} error: must be train or test, not {}'.format(opt.trainer_mode))
    
    print('finish')


def diffusion_init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  
        mp.set_start_method("spawn", force=True) 
    rank = int(os.environ["RANK"])  
    num_gpus = torch.cuda.device_count() 
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  



if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    universal_main(opt=opt)


