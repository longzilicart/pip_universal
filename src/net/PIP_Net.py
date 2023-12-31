# modified from the original Restormer https://github.com/swz30/Restormer
# add pip and p2f in the skip connection.


import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from net.PIP import * # pip module


class PIPNet_Restormer_onskip_inter(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        # ++++ RAP settings as plug-in-and-play module ++++
        decoder = True,
        use_detask_label = True,
        use_detask_prompt = True,
        use_CLIP_prompt = False,
        use_SAM_prompt = False,
        use_degradation_sensor = False,
        high_prompt_dim = 1, 
        low_prompt_dims = [64, 128, 320],
        prompt_interaction_mode = 'promptir', # convgate # promptir_gate # promptir
        degradation_num = 5,
        low_prompt_sizes = [64, 32, 16] 
        ):
        super(PIPNet_Restormer_onskip_inter, self).__init__()
        # promptir

        # ++++++++++++ RAP settings ++++++++++++
        self.decoder = decoder
        use_detask_label = use_detask_label
        use_detask_prompt, use_CLIP_prompt, use_SAM_prompt = use_detask_prompt, use_CLIP_prompt, use_SAM_prompt
        use_degradation_sensor = use_degradation_sensor

        # no need anymore, debug only
        lin_dims = [96, 192, 384]
        # >>>>>>>>> analysis >>>>>>>>
        self.hook_outputs = {}  
        self.use_hooks = False
        # >>>>>>>>> analysis >>>>>>>>

        if self.decoder:
            self.prompt1 = PromptInPrompt(prompt_dim=high_prompt_dim,task_classes=degradation_num,
                            prompt_size = low_prompt_sizes[0], lin_dim = 96, 
                            low_prompt_dim=low_prompt_dims[0],
                            use_detask_label=use_detask_label,
                            use_detask_prompt=use_detask_prompt, use_SAM_prompt=use_SAM_prompt, use_CLIP_prompt=use_CLIP_prompt,
                            use_degradation_sensor=use_degradation_sensor,)
            self.prompt2 = PromptInPrompt(prompt_dim=high_prompt_dim,task_classes=degradation_num,prompt_size = low_prompt_sizes[1], lin_dim = 192, 
                            low_prompt_dim=low_prompt_dims[1],
                            use_detask_label=use_detask_label,
                            use_detask_prompt=use_detask_prompt, use_SAM_prompt=use_SAM_prompt, use_CLIP_prompt=use_CLIP_prompt,
                            use_degradation_sensor=use_degradation_sensor,)
            self.prompt3 = PromptInPrompt(prompt_dim=high_prompt_dim,task_classes=degradation_num,prompt_size = low_prompt_sizes[2], lin_dim = 384, 
                            low_prompt_dim=low_prompt_dims[2],
                            use_detask_label=use_detask_label,
                            use_detask_prompt=use_detask_prompt, use_SAM_prompt=use_SAM_prompt, use_CLIP_prompt=use_CLIP_prompt,
                            use_degradation_sensor=use_degradation_sensor,)

        # ============= Restormer settings ================
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)      
        # self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        # self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        self.chnl_reduce3 = nn.Conv2d(int(dim*2**3),int(dim*2**2),kernel_size=1,bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock_Restormer(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock_Restormer(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock_Restormer(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock_Restormer(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1) + 192, int(dim*2**2), kernel_size=1, bias=bias)

        # >>>>>>>>> prompt interaction >>>>>>>>>
        interaction_mode = prompt_interaction_mode #
        if interaction_mode in ['promptir', 'promptir_gate', 'pip_cross', 'pip_cross_topm']:
            interaction_opt = {
                'feat_dim': int(dim*2**2), 'prompt_dim': low_prompt_dims[2], 'head': heads[2], 
                'ffn_expansion_factor': ffn_expansion_factor, 
                'LayerNorm_type': LayerNorm_type,
                'bias': bias, }
        elif interaction_mode in ['convgate']:
            interaction_opt = {
                    'in_feat': low_prompt_dims[2],
                    'out_feat': int(dim*2**2),}
        else:
            raise ValueError(f"Unknown interaction mod: {interaction_mode}")
        self.low_prompt_interaction_level3 = PromptToFeature(interaction_mode = interaction_mode, interaction_opt = interaction_opt)
        # >>>>>>>>> prompt interaction >>>>>>>>>


        self.decoder_level3 = nn.Sequential(*[TransformerBlock_Restormer(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)

        # >>>>>>>>> prompt interaction >>>>>>>>        
        if interaction_mode in ['promptir', 'promptir_gate', 'pip_cross', 'pip_cross_topm']:
            interaction_opt = {
                'feat_dim': int(dim*2**1), 'prompt_dim': low_prompt_dims[1], 'head': heads[2], 
                'ffn_expansion_factor': ffn_expansion_factor, 
                'LayerNorm_type': LayerNorm_type,
                'bias': bias, }
        elif interaction_mode in ['convgate']:
            interaction_opt = {
                    'in_feat': low_prompt_dims[1],
                    'out_feat': int(dim*2**1),}
        self.low_prompt_interaction_level2 = PromptToFeature(interaction_mode = interaction_mode, interaction_opt = interaction_opt)
        # >>>>>>>>> prompt interaction >>>>>>>>>
        self.decoder_level2 = nn.Sequential(*[TransformerBlock_Restormer(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        # >>>>>>>>> prompt interaction >>>>>>>>>
        if interaction_mode in ['promptir', 'promptir_gate', 'pip_cross', 'pip_cross_topm']:
            interaction_opt = {
                'feat_dim': int(dim*2**0), 'prompt_dim': low_prompt_dims[0], 'head': heads[2], 
                'ffn_expansion_factor': ffn_expansion_factor, 
                'LayerNorm_type': LayerNorm_type,
                'bias': bias, }
        elif interaction_mode in ['convgate']:
            interaction_opt = {
                    'in_feat': low_prompt_dims[0],
                    'out_feat': int(dim*2**0),}
        self.low_prompt_interaction_level1 = PromptToFeature(interaction_mode = interaction_mode, interaction_opt = interaction_opt)
        # >>>>>>>>> prompt interaction >>>>>>>>>


        self.decoder_level1 = nn.Sequential(*[TransformerBlock_Restormer(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock_Restormer(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # 临时
        self.dim = dim

    def forward(self, inp_img, noise_emb = None, degradation_class = None):
        '''
        if label is not None
            we use it as one hot label to select prompt
        '''

        # encoder
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)
        
        # ===== pip integrated in skip connection in decoder ====
        if self.decoder:
            dec3_param, dec3_P = self.prompt3(out_enc_level3, degradation_class)
            if self.use_hooks:
                self.hook_outputs['dec3_param'] = dec3_param.detach()
            out_enc_level3 = self.low_prompt_interaction_level3(out_enc_level3, dec3_param)
        latent = self.chnl_reduce3(latent) 

        # skip connection (original network)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        
        if self.decoder:
            dec2_param, dec2_P = self.prompt2(out_enc_level2, degradation_class)
            if self.use_hooks:
                self.hook_outputs['dec2_param'] = dec2_param.detach()
            out_enc_level2 = self.low_prompt_interaction_level2(out_enc_level2, dec2_param)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        if self.decoder:   
            dec1_param, dec1_P = self.prompt1(out_enc_level1, degradation_class)
            if self.use_hooks:
                self.hook_outputs['dec1_param'] = dec1_param.detach()
            out_enc_level1 = self.low_prompt_interaction_level1(out_enc_level1, dec1_param)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        # prompt parameters
        return out_dec_level1, [dec3_P, dec2_P, dec1_P]


    # =========== regulaztion from my promptir module ============
    def param_regulaztion_loss(self, reg_opt = 'cosine', 
                               reg_which = ['level_1', 'level_2', 'level_3'],
                               reg_which_ratio = [1., 1., 1.],
                               ):
        
        assert reg_opt in ['3d', '2d', 'cosine', 'angle']
        param_reg_loss = torch.tensor(0.0).cuda()
        if 'level_1' in reg_which:
            param_reg_loss += self.prompt1.get_detask_prompt_reg_loss(reg_opt) * reg_which_ratio[0]
        if 'level_2' in reg_which:
            param_reg_loss += self.prompt2.get_detask_prompt_reg_loss(reg_opt) * reg_which_ratio[1]
        if 'level_3' in reg_which:
            param_reg_loss += self.prompt3.get_detask_prompt_reg_loss(reg_opt) * reg_which_ratio[2]
        return param_reg_loss












if __name__ == '__main__':

    # have a try
    decoder = True
    use_detask_label = True
    use_detask_prompt = True
    use_CLIP_prompt = False
    # use_CLIP_prompt = True
    use_SAM_prompt = False
    use_degradation_sensor = False
    high_prompt_dim = 1     
    net = PIPNet_Restormer_onskip_inter(inp_channels=3, out_channels=3,
                        decoder = decoder,
                        use_detask_label = use_detask_label,
                        use_detask_prompt = use_detask_prompt,
                        use_CLIP_prompt = use_CLIP_prompt,
                        use_SAM_prompt = use_SAM_prompt,
                        use_degradation_sensor = use_degradation_sensor,
                        high_prompt_dim = high_prompt_dim, )

    x = torch.randn((1, 3, 256, 256))
    y, _ = net(x, degradation_class = torch.tensor(0))
    print(y.shape)








