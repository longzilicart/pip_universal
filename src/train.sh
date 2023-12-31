
# -- logger path -- 
checkpoint_root='' 
tensorboard_root='' 
wandb_root='' 

# -- dataset path -- similar to the promptir  
dataset_path='DATA' # such as DATA, can be any where


# /data_dir # include the list for hazy noisy rainy (txt inside)
#     /hazy 
#     /noisy
#     /rainy
# /data
#     /Train
#         /Dehaze (/original /synthetic)
#         /Denoise (images)
#         /Derain (/rainy /gt)
#         /Enhance (/high /low)
#         /Deblur (/train /test /valid)
#     /test
#          .... similar


# training demo
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch \
--master_port 19954 \
--nproc_per_node 8 \
Main_trainer.py \
--trainer_mode 'train' --network PIPNet_Restormer_onskip_inter --high_reg_loss angle \
--checkpoint_root $checkpoint_root \
--tensorboard_root $tensorboard_root \
--checkpoint_dir PIPNet_Restormer_onskip_inter \
--tensorboard_dir  PIPNet_Restormer_onskip_inter \
--dataset_path $dataset_path \
--batch_size 6 --lr 36e-5 --num_workers 4 --log_interval 100 \
--de_type denoise_15 denoise_25 denoise_50 derain dehaze deblur lowlight \
--use_wandb --wandb_project '[universal]' --wandb_root $wandb_root \



