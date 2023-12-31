from Basic_Trainer import *
from dataclasses import dataclass
from data_provider.universal_dataset import *
from torchvision.utils import save_image




# PIPNet_Trainer 
class PIPNet_Trainer(Restoration_Trainer_Basic):
    def __init__(self, net, opt = None, degration_num = 5):     
        super(PIPNet_Trainer, self).__init__()
        self.degradation_num = degration_num

        assert net is not None and opt is not None
        self.net = net
        self.opt = opt
        self.prepare_args() # dataset file path
        # define the dataset 
        self.train_dataset = PromptTrainDataset(opt, noise_combine = True)
        # simple validation, support any dataset
        self.create_val_datasets()
        if self.opt.is_test:
            self._init_lpips()

    def prepare_args(self,):
        opt = self.opt
        # train path
        self.opt.data_file_dir = os.path.join(opt.dataset_path, 'data_dir')
        self.opt.denoise_dir = os.path.join(opt.dataset_path, 'data/Train/Denoise/')
        self.opt.derain_dir = os.path.join(opt.dataset_path, 'data/Train/Derain/')
        self.opt.dehaze_dir = os.path.join(opt.dataset_path, 'data/Train/Dehaze/')
        self.opt.gopro_dir = os.path.join(opt.dataset_path, 'data/Train/GOPRO_deblur/')
        self.opt.lowlight_dir = os.path.join(opt.dataset_path, 'data/Train/Enhance')
        
        @dataclass
        class TestOptions:
            mode: int = 3  # TODO no use 
            denoise_path: str = os.path.join(opt.dataset_path,"data/test/denoise/")  
            derain_path: str = os.path.join(opt.dataset_path, "data/test/derain/")  
            dehaze_path: str = os.path.join(opt.dataset_path, "data/test/dehaze/")  
            gopro_dir: str = os.path.join(opt.dataset_path, "data/test/GOPRO_deblur/")
            lowlight_dir: str = os.path.join(opt.dataset_path, 'data/test/enhance')
            output_path: str = "output/"  
        self.testopt = TestOptions()


    def prepare_dataset(self, ):
        '''prepare train and val dataset here'''
        pass


    def init_adam_optimizer(self, net):
        self.optimizer = torch.optim.Adam(net.parameters(), lr = self.opt.lr, betas = (self.opt.beta1, self.opt.beta2)) 
        self.step_optimizer = LinearWarmupCosineAnnealingLR(optimizer=self.optimizer,warmup_epochs=10, max_epochs=150,warmup_start_lr=self.opt.lr/10)


    def init_adamw_optimizer(self, net):
        # initialize adamw 
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2), weight_decay=self.opt.weight_decay,)
        self.step_optimizer = LinearWarmupCosineAnnealingLR(optimizer=self.optimizer,warmup_epochs=10, max_epochs=120, warmup_start_lr=self.opt.lr/10)


    # fit pip network
    def fit(self, ):
        opt = self.opt
        self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend = 'nccl')
        device = torch.device('cuda', opt.local_rank)
        self.cards = torch.distributed.get_world_size()
        # resume model
        if self.opt.resume is True:
            self.resume()
        else: 
            pass

        # network to device, DDP
        self.net = self.net.to(device)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net,
                                                device_ids = [opt.local_rank],
                                                output_device = opt.local_rank,
                                                find_unused_parameters=True)
                                                # find_unused_parameters=True
        
        # start my logger 
        wandb_dir = opt.tensorboard_dir
        self.logger = Longzili_Logger(
            log_name = str(wandb_dir),
            project_name = opt.wandb_project,
            config_opt = opt,
            checkpoint_root_path = opt.checkpoint_root,
            tensorboard_root_path = opt.tensorboard_root,
            wandb_root_path = opt.wandb_root,
            use_wandb = opt.use_wandb,
            log_interval = opt.log_interval,)

        # define the dataset
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  sampler = train_sampler,
                                  pin_memory = True,
                                  drop_last = opt.drop_last,
                                  ) 

        # init and resume optimizer
        if self.opt.network in ['restormer', 'promptir']:
            self.init_adamw_optimizer(self.net)
        else:
            self.init_adamw_optimizer(self.net)
        if self.opt.resume_opt is True:
            self.resume_opt()
            myprint('resume{}'.format(self.epoch))

        # start training loop
        start_epoch = self.epoch
        world_size = torch.distributed.get_world_size()
        for self.epoch in range(start_epoch, opt.epochs):
            if self.opt.local_rank == 0:
                myprint('start trining epoch:{}'.format(self.epoch))
            self.train_loader.sampler.set_epoch(self.epoch)
            info_dict = {
                'epoch': self.epoch,
                'batch_size': self.opt.batch_size,
                'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                'lr_per_sample': self.optimizer.state_dict()['param_groups'][0]['lr']/opt.batch_size/world_size,
                }
            self.logger.log_info_dict(info_dict)
            self.train()
            self.val()
            torch.cuda.empty_cache()
            self.step_optimizer.step()
            if self.opt.local_rank == 0:
                self.save_model()
                self.save_opt()

    def train(self, ):
        '''train pip module'''
        opt = self.opt
        alpha = 0.002
        beta = 1
        task_classes = self.degradation_num
    
        if opt.high_reg_loss not in ['None', 'none', '']: # Default angle
            if 'bottle' in self.opt.network:
                reg_opt, reg_which, reg_which_ratio = opt.high_reg_loss, ['level_1', 'level_2', 'level_3', 'level4'], [1., 1., 1., 1.],
            elif 'Restormer' in self.opt.network:
                reg_opt, reg_which, reg_which_ratio = opt.high_reg_loss, ['level_1', 'level_2', 'level_3'], [1., 1., 1.],
            elif 'NAF' in self.opt.network:
                reg_opt, reg_which, reg_which_ratio = opt.high_reg_loss, ['level_1', 'level_2', 'level_3', 'level4'], [1., 1., 1., 1.],
            else:
                raise NotImplementedError


        # train the model
        self.net.train()
        pbar = tqdm.tqdm(self.train_loader, ncols = 60,
                        disable=not self.opt.local_rank == 0)
        for i, data in enumerate(pbar):
            # if i>101: # debug
            #     break
            ([clean_name, de_id], degrad_patch, clean_patch) = data
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            restored, predictions = self.net(degrad_patch, degradation_class = de_id)

            if 'pip' in opt.network.lower():
                # stable training avoid large loss
                pixel_loss = self.scale_pixel_loss(restored, clean_patch, mode = 'sml1')
            
                # reg loss
                if opt.high_reg_loss not in ['None', 'none', '']:
                    param_reg_loss = self.net.module.param_regulaztion_loss(reg_opt = reg_opt, reg_which = reg_which, reg_which_ratio = reg_which_ratio)
                else:
                    param_reg_loss = torch.tensor(0)

                pred_loss1 = pred_loss2 = pred_loss3 = torch.tensor(0)
                pred_loss = pred_loss1 + pred_loss2 + pred_loss3
                loss = pixel_loss + param_reg_loss * alpha + pred_loss * beta
            
            else:
                raise NotImplementedError(f"not support network:{opt.network} for pip training")

            # --- backward ---
            self.optimizer.zero_grad()
            self.reduce_loss(loss).backward()
            self.optimizer.step()

            # --- get matrix ---
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            # --- logger ----
            self.logger.tick()
            iter_log = {'loss': loss,
                        'pixel_loss': pixel_loss,
                        'param_reg_loss': param_reg_loss,
                        'psnr': temp_psnr,
                        'ssim': temp_ssim,
                        'pred_loss': pred_loss,
                        'pred_loss1': pred_loss1,
                        'pred_loss2': pred_loss2,
                        'pred_loss3': pred_loss3,
                    }
            self.logger.log_scalar_dict(iter_log, training_stage='train')        

        # log epoch information 
        self.logger.log_scalar(force=True, log_type='epoch', training_stage='train')
        epoch_image = {
                'degrad_path': degrad_patch,
                'clean_patch': clean_patch,
                'restored_patch': restored,
                }
        self.logger.log_image_dict(epoch_image, log_type='epoch', force=True, training_stage = 'train_image')    


    # validation no need for faster training
    @torch.no_grad()
    def val(self,):
        myprint('a simple but ugly validation func')

        self.task_dedict = {'denoise_bsd_sigma15': 0, 'denoise_bsd_sigma25': 0, 
                          'denoise_bsd_sigma50': 0, 
                        'derain_R100_set': 1, 'dehaze_SOTSout_set': 2, 
                        'deblur' : 3, 'enhance': 4}
        val_batch = 1
        self.net.eval()
        for (dataset_name, dataset) in self.test_dastaset_list:
            a_val_loader = self.create_val_dataloader(dataset, batch_size = val_batch, num_workers=0, distributed=True)
            pbar = tqdm.tqdm(a_val_loader,ncols = 60,disable=not self.opt.local_rank == 0)
            for i, data in enumerate(pbar):
                ([degraded_name], degrad_patch, clean_patch) = data
                degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
                if 'pip' in self.opt.network.lower():
                    de_id = self.task_dedict[dataset_name]
                    de_id = torch.tensor(de_id).repeat(val_batch) 
                    restored, predictions = self.net(degrad_patch, degradation_class = de_id)
                else: 
                    restored = self.net(degrad_patch)

                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                iter_log = {
                            f'{dataset_name}_psnr': temp_psnr,
                            f'{dataset_name}_ssim': temp_ssim,}
                self.logger.log_scalar_dict(iter_log, training_stage='val')
            
                if self.opt.is_test:
                    self.trainer_image_saver(degrad_patch, restored, clean_patch, degraded_name, dataset_name)    

            # log validation information for each validation dataset 
            val_image_info = {'degrad_patch': degrad_patch,
                        'restored': restored,
                        'clean_patch': clean_patch}
            self.logger.log_scalar(force=True, log_type='epoch', training_stage = 'val')
            self.logger.log_image_dict(val_image_info, log_type='epoch', force=True, training_stage = 'val_image') 


    def create_val_datasets(self,):
        testopt = self.testopt
        # 1.1 denoise    
        denoise_base_path = testopt.denoise_path
        denoise_splits = ["bsd68/original/"] 
        for name in denoise_splits:
            testopt.denoise_path = os.path.join(denoise_base_path, name)
            denoise_bsd_sigma15 = DenoiseTestDataset(testopt)
            denoise_bsd_sigma15.set_sigma(15)
            denoise_bsd_sigma25 = DenoiseTestDataset(testopt)
            denoise_bsd_sigma25.set_sigma(25)
            denoise_bsd_sigma50 = DenoiseTestDataset(testopt)
            denoise_bsd_sigma50.set_sigma(50)
        # 1.2 derain and dehazing
        derain_base_path = testopt.derain_path
        derain_splits = ["Rain100L/"]
        for name in derain_splits:
            myprint('generate rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_R100_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15, )
            derain_R100_set.set_dataset('derain')
        # 1.3 dehazing
        dehaze_splits = ["outdoor/"]
        dehaze_base_path = testopt.dehaze_path
        testopt.dehaze_path = os.path.join(dehaze_base_path, dehaze_splits[0])
        dehaze_SOTSout_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
        dehaze_SOTSout_set.set_dataset("dehaze")

        # 1.4 debluring 
        deblur_gopro_set = DeblurTestDataset(testopt, is_val=True)

        # 1.5 lowlight enahcnement
        enhance_lol_set = LowLightTestDataset(testopt)

        test_dataset_list = []
        if 'denoise_15' in self.opt.de_type:
            test_dataset_list.extend([('denoise_bsd_sigma15', denoise_bsd_sigma15),
                            ('denoise_bsd_sigma25', denoise_bsd_sigma25),
                            ('denoise_bsd_sigma50', denoise_bsd_sigma50)])
        if 'derain' in self.opt.de_type:
            test_dataset_list.append(('derain_R100_set', derain_R100_set))
        if 'dehaze' in self.opt.de_type:
            test_dataset_list.append(('dehaze_SOTSout_set', dehaze_SOTSout_set))
        if 'deblur' in self.opt.de_type:
             test_dataset_list.append(('deblur', deblur_gopro_set))
        if 'lowlight' in self.opt.de_type:
            test_dataset_list.append(('enhance', enhance_lol_set))
        myprint('finish perparing test dataset')
        self.test_dastaset_list = test_dataset_list
        myprint(test_dataset_list)


    def create_val_dataloader(self, dataset, batch_size, num_workers, distributed=False):
        if distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            val_sampler = None
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler)
        return loader

    def trainer_image_saver(self, degrad_img, restored_img, clean_img, degraded_name, dataset_name):
        '''
        if save image in the trainer
        args:
            degraded_name: name of a specific degraded image in the daatset
            dataset_name: dataset name
            clean_patch: clean image
            image_save_path: path to save the image
        '''

        # calculate PSNR, SSIM, lpips and more 
        temp_psnr, temp_ssim, _ = compute_psnr_ssim(restored_img, clean_img)
        temp_lpips = self.perceptural_similarity_fn(restored_img.cpu(), clean_img.cpu())
        img_name = f"{degraded_name}_psnr_{temp_psnr:.4f}_ssim_{temp_ssim:.4f}_lpipss_{temp_lpips:.4f}.png"
        if self.opt.tester_save_path is not None:
            save_dir = os.path.join(self.opt.tester_save_path, self.opt.checkpoint_dir, dataset_name)
        else:
            raise ValueError("...self.opt.tester_save_path is None...")
        # save image 
        os.makedirs(save_dir, exist_ok=True)
        self._trainer_save_image(degrad_img, os.path.join(save_dir, img_name))
        self._trainer_save_image(restored_img, os.path.join(save_dir, img_name))
        self._trainer_save_image(clean_img, os.path.join(save_dir, img_name))

    def trainer_stats_saver(self, degrad_img, restored_img, clean_img,  degraded_name, dataset_name, image_save_path):
        pass

    @staticmethod
    def _trainer_save_image(image, path):
        save_image(image, path)

    def _init_lpips(self):
        import lpips
        self.perceptural_similarity_fn = lpips.LPIPS(net='vgg') 
        # closer to "traditional" perceptual loss,
        # usage
        # d = loss_fn_alex(img0, img1)
 



















class PIPNet_Sensor_Trainer(PIPNet_Trainer):
    '''
    trainer for degradation-aware sensor
    '''
    def __init__(self, net, opt = None, degration_num = 5):     
        super(PIPNet_Sensor_Trainer, self).__init__(net, opt=opt, degration_num=degration_num)
        self.degradation_num = degration_num
        self.criterion = nn.CrossEntropyLoss()

    def init_adam_optimizer(self, net):
        self.optimizer = torch.optim.Adam(net.parameters(), lr = self.opt.lr, betas = (self.opt.beta1, self.opt.beta2)) # TODO 这里没有操作
        self.step_optimizer = StepLR(self.optimizer, step_size = self.opt.step_size, gamma=self.opt.step_gamma)

    # fit the degradation aware sensor
    def fit(self, ):
        opt = self.opt
        self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend = 'nccl')
        device = torch.device('cuda', opt.local_rank)
        self.cards = torch.distributed.get_world_size()
        # resume model
        if self.opt.resume is True:
            self.resume()
        else: 
            pass

        # network to device, DDP
        self.net = self.net.to(device)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net,
                                                device_ids = [opt.local_rank],
                                                output_device = opt.local_rank,
                                                find_unused_parameters=True)

        # start my logger 
        wandb_dir = opt.tensorboard_dir
        self.logger = Longzili_Logger(
            log_name = str(wandb_dir),
            project_name = opt.wandb_project,
            config_opt = opt,
            checkpoint_root_path = opt.checkpoint_root,
            tensorboard_root_path = opt.tensorboard_root,
            wandb_root_path = opt.wandb_root,
            use_wandb = opt.use_wandb,
            log_interval = opt.log_interval,)

        # define the dataset
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  sampler = train_sampler,
                                  pin_memory = True,
                                  drop_last = opt.drop_last,
                                  ) 
        
        # self.init_adam_optimizer(self.net)
        self.init_adamw_optimizer(self.net)
        if self.opt.resume_opt is True:
            self.resume_opt()
            myprint('resume{}'.format(self.epoch))

        # start training loop
        start_epoch = self.epoch
        world_size = torch.distributed.get_world_size()
        for self.epoch in range(start_epoch, opt.epochs):
            if self.opt.local_rank == 0:
                myprint('start trining epoch:{}'.format(self.epoch))
            self.train_loader.sampler.set_epoch(self.epoch)
            info_dict = {
                'epoch': self.epoch,
                'batch_size': self.opt.batch_size,
                'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                'lr_per_sample': self.optimizer.state_dict()['param_groups'][0]['lr']/opt.batch_size/world_size,
                }
            self.logger.log_info_dict(info_dict)
            self.train()
            self.val()
            torch.cuda.empty_cache()
            self.step_optimizer.step()
            if self.opt.local_rank == 0:
                self.save_model()
                self.save_opt()



    def train(self, ):
        '''
        the model should contain the t
        '''
        task_classes = self.degradation_num
        total = 141419
        # balance data
        # class_weight = {
        #     0: (1/42696) * (total) / 2.0,
        #     1: (1/24000) * (total) / 2.0,
        #     2: (1/72135) * (total) / 2.0,
        #     3: (1/2103) * (total) / 2.0,
        #     4: (1/485) * (total) / 2.0,
        # }
        # class_weight=class_weight

        self.net.train()
        pbar = tqdm.tqdm(self.train_loader, ncols = 60,
                        disable=not self.opt.local_rank == 0)
        for i, data in enumerate(pbar):
            # if i>101: # debug
            #     break
            ([clean_name, de_id], degrad_patch, clean_patch) = data
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            de_id = de_id.cuda()

            prob = self.net(degrad_patch)
            loss = self.criterion(prob.cpu(), de_id.cpu())

            # --- backward ---
            self.optimizer.zero_grad()
            self.reduce_loss(loss).backward()
            self.optimizer.step()

            _, predicted = torch.max(prob, 1)
            correct_predictions = torch.eq(predicted, de_id).sum().item()
            accuracy = correct_predictions / de_id.size(0)

            # --- logger ----
            self.logger.tick()
            iter_log = {'train/loss': loss,
                        'train/accuracy': accuracy,
                    }
            self.logger.log_scalar_dict(iter_log, training_stage='train')       
        self.logger.log_scalar(force=True, log_type='epoch', training_stage='train')

    @torch.no_grad()
    def val(self,):
        myprint('a simple but ugly validation func')
        self.task_dedict = {'denoise_bsd_sigma15': 0, 'denoise_bsd_sigma25': 0, 
                          'denoise_bsd_sigma50': 0, 
                        'derain_R100_set': 1, 'dehaze_SOTSout_set': 2, 
                        'deblur' : 3, 'enhance': 4} 
        
        val_batch = 1
        self.net.eval()
        for (dataset_name, dataset) in self.test_dastaset_list:
            a_val_loader = self.create_val_dataloader(dataset, batch_size = val_batch, num_workers=0, distributed=True)
            pbar = tqdm.tqdm(a_val_loader,ncols = 60,disable=not self.opt.local_rank == 0)
            for i, data in enumerate(pbar):
                ([degraded_name], degrad_patch, clean_patch) = data
                prob_list = []
                for _ in range(3):
                    degrad_crop = self._random_crop(degrad_patch, (self.opt.patch_size, self.opt.patch_size))
                    p = self.net(degrad_crop)
                    prob_list.append(p)
                prob = torch.cat(prob_list, dim=0).mean(dim=0, keepdims=True) # accuracy on batch
                
                de_id = torch.tensor([self.task_dedict[dataset_name]]).cuda()
                _, predicted = torch.max(prob, 1)
                correct_predictions = torch.eq(predicted, de_id).sum().item()
                accuracy = correct_predictions / de_id.size(0)

                iter_log = {f'{dataset_name}_accuracy': accuracy,}
                self.logger.log_scalar_dict(iter_log, training_stage='val')       
            # log validation information for each validation dataset 

            self.logger.log_scalar(force=True, log_type='epoch', training_stage = 'val')



    @staticmethod
    def update_accuracy_and_errors(stats, prob, labels):
        raise NotImplementedError('draft')
        _, predicted = torch.max(prob, 1)
        correct = (predicted == labels).float().sum()
        total = labels.size(0)
        accuracy = correct / total
        stats['correct'] += correct
        stats['total'] += total
        for i in range(prob.size(1)):
            class_wrong = (predicted == i).float().sum() - (labels == i).float().sum()
            stats['class_wrong'][i] += class_wrong

    @staticmethod
    def _random_crop(img, size):
        assert len(img.shape) == 4
        assert len(size) == 2
        b, c, h, w = img.shape
        y_min = torch.randint(0, h - size[0], (b,))
        x_min = torch.randint(0, w - size[1], (b,))

        cropped_img = img[
            :, :, y_min:y_min + size[0], x_min:x_min + size[1]
        ]
        return cropped_img





if __name__ == '__main__':
    pass



