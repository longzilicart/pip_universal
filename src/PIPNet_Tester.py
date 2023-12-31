from PIPNet_Trainer import *



class PIPNet_Tester(PIPNet_Trainer):
    def __init__(self, net, opt = None, degration_num = 5):     
        super(PIPNet_Tester, self).__init__(net, opt=opt, degration_num=degration_num)
        
        self.degradation_num = degration_num
        assert net is not None and opt is not None
        self.net = net
        self.opt = opt
        self.prepare_args()

        # define the dataset 
        self.train_dataset = PromptTrainDataset(opt, noise_combine = True)
        self.create_val_datasets()

        if self.opt.is_test:
            self._init_lpips() 

    def tester_resume(self):
        '''resume training
        '''
        if self.opt.net_checkpath is not None:
            self.tester_load_model(strict=True)
            myprint('finish loading model')
        else:
            raise ValueError('opt.net_checkpath not provided')

    # reconstruct load model function 
    def tester_load_model(self, strict=False):
        '''basic load model'''
        net_checkpath = self.opt.net_checkpath
        try:
            net_checkpoint = torch.load(net_checkpath, map_location = 'cpu')['params']
        except Exception as err:
            net_checkpoint = torch.load(net_checkpath, map_location = 'cpu')
        
        if strict:
            self.net.load_state_dict(net_checkpoint, strict=strict) # strict = False
        else:
            self.load_partial_state_dict(net_checkpoint)
        # epoch not need
        myprint('finish loading network')

    
    def fit(self, ):
        opt = self.opt
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend = 'nccl')
        device = torch.device('cuda', opt.local_rank)
        self.cards = torch.distributed.get_world_size()
        # resume model

        if self.opt.network == "promptir_test":
            pass
        else:
            self.tester_resume() 
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
            project_name = opt.wandb_project + "_test", # new project for test only
            config_opt = opt,
            checkpoint_root_path = opt.checkpoint_root,
            tensorboard_root_path = opt.tensorboard_root,
            wandb_root_path = opt.wandb_root,
            use_wandb = opt.use_wandb,
            log_interval = opt.log_interval,)

        world_size = torch.distributed.get_world_size()
        self.test()
        torch.cuda.empty_cache()

    # tester for any multi-task model
    @torch.no_grad()
    def test(self,):       
        self.net.eval() 
        self.task_dedict = {'denoise_bsd_sigma15': 0, 'denoise_bsd_sigma25': 0, 
                          'denoise_bsd_sigma50': 0, 
                        'derain_R100_set': 1, 'dehaze_SOTSout_set': 2, 
                        'deblur' : 3, 'enhance': 4,
                        }

        val_batch = 1

        # dataloaders = {}
        for (dataset_name, dataset) in self.test_dastaset_list:

            a_val_loader = self.create_val_dataloader(dataset, batch_size = val_batch, num_workers=0, distributed=True)
            pbar = tqdm.tqdm(a_val_loader,ncols = 60,disable=not self.opt.local_rank == 0)
            for i, data in enumerate(pbar):

                ([degraded_name], degrad_patch, clean_patch) = data
                degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
                if 'RAP' in self.opt.network:
                    de_id = self.task_dedict[dataset_name]
                    if isinstance(de_id, int):
                        de_id = torch.tensor(de_id).repeat(val_batch)
                    else: # tensor
                        de_id = de_id
                    restored, predictions = self.net(degrad_patch, degradation_class = de_id)
                elif 'airnet' in self.opt.network:
                    restored = self.net(x_query=degrad_patch, x_key=degrad_patch)
                else: 
                    # other networks
                    restored = self.net(degrad_patch)


                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                iter_log = {
                            f'{dataset_name}_psnr': temp_psnr,
                            f'{dataset_name}_ssim': temp_ssim,}
                self.logger.log_scalar_dict(iter_log, training_stage='val')
            
                if self.opt.is_test:
                    if self.opt.is_save_image:
                        # TESTER (path = {iamge_save_path}/{opt.checkpointname}/dataset_name/degraded_name_psnr_ssim.png})
                        if dataset_name == "denoise_aapm_sigma25":
                            self.trainer_image_saver(degrad_patch.mean(dim=1,keepdim=True), restored.mean(dim=1,keepdim=True), clean_patch.mean(dim=1,keepdim=True), degraded_name, dataset_name)
                        else:
                            self.trainer_image_saver(degrad_patch, restored, clean_patch, degraded_name, dataset_name)
                    else:   
                        pass 


            self.logger.log_scalar(force=True, log_type='epoch', training_stage = 'val')


    def trainer_image_saver(self, degrad_img, restored_img, clean_img,  degraded_name, dataset_name):

        if self.opt.tester_save_name == '':
            net_name = self.opt.checkpoint_dir
        else:
            net_name = self.opt.tester_save_name
        if self.opt.tester_save_path is not None:
            save_dir = os.path.join(self.opt.tester_save_path, net_name, dataset_name)
        else:
            raise ValueError("...self.opt.tester_save_path is None...")
        # save image 
        os.makedirs(save_dir, exist_ok=True)

        # [degrad image]
        temp_psnr, temp_ssim, _ = compute_psnr_ssim(degrad_img, clean_img)
        temp_lpips = self.perceptural_similarity_fn(degrad_img.cpu(), clean_img.cpu())
        temp_psnr, temp_ssim, temp_lpips = temp_psnr, temp_ssim, temp_lpips.detach().cpu().item()#.numpy()
        img_name = f"{net_name}_{degraded_name}_degrad_psnr_{temp_psnr:.4f}_ssim_{temp_ssim:.4f}_lpipss_{temp_lpips:.4f}.png"
        self._trainer_save_image(degrad_img, os.path.join(save_dir, img_name))

        # [restore image]
        temp_psnr, temp_ssim, _ = compute_psnr_ssim(restored_img, clean_img)
        temp_lpips = self.perceptural_similarity_fn(restored_img.cpu(), clean_img.cpu())
        temp_psnr, temp_ssim, temp_lpips = temp_psnr, temp_ssim, temp_lpips.detach().cpu().item()#.numpy()
        img_name = f"{net_name}_{degraded_name}_restore_psnr_{temp_psnr:.4f}_ssim_{temp_ssim:.4f}_lpipss_{temp_lpips:.4f}.png"
        self._trainer_save_image(restored_img, os.path.join(save_dir, img_name))

        # [clean image]
        img_name = f"{net_name}_{degraded_name}_clean_psnr_{temp_psnr:.4f}_ssim_{temp_ssim:.4f}_lpipss_{temp_lpips:.4f}.png"
        self._trainer_save_image(clean_img, os.path.join(save_dir, img_name))
        

    def trainer_stats_saver(self, degrad_img, restored_img, clean_img,  degraded_name, dataset_name, image_save_path):
        pass
        # not provide yet, plan to use wandb and my logger directly

    @staticmethod
    def _trainer_save_image(image, path):
        # use torchvision.utils.save_image to save PNG images
        save_image(image, path)

    def _init_lpips(self):
        import lpips
        self.perceptural_similarity_fn = lpips.LPIPS(net='vgg') 



