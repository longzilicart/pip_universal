# prompt in prompt learning
# include
# 1. basic module in the framework
# 2. basic prompt in prompt learning module
# 3. pip module in the paper

import sys
sys.path.append("..")
from net.PIP_utils import *




# ================= PIP learning module ================
# ================= PIP learning module ================
# ================= PIP learning module ================
# ================= PIP learning module ================

class Basic_prompt(nn.Module):
    '''
    basic module to build pip module.
    intro:
        1. different kind of regularization function
            regulaztion_loss_promptlen_3d
            regulaztion_loss_promptlen_2d
            regulaztion_loss_promptlen_cosine
            regulaztion_loss_promptlen_angle: theta>angle
        2. prompt_shape (5D):
            (1, prompt_len, prompt_dim, shape, shape) 
            (B, prompt_len, prompt_dim, sahpe, shape)
    '''
    def __init__(self,):
        super(Basic_prompt, self).__init__()
        self.lambda_orthogonal = 0.01

    def _regulaztion_loss_promptlen_3d(self, prompt_param):
        ''' ortho on prompt_len, directly on 3D tensor'''
        reshaped_param = prompt_param.permute(3, 4, 1, 2, 0).reshape(-1, self.prompt_param.size(1), self.prompt_param.size(2))
        loss = self._batch_orthogonal_loss(reshaped_param)
        return loss
        
    @staticmethod
    def _batch_orthogonal_loss(W):
        lambda_orthogonal = 0.01
        WT_W_minus_I = torch.bmm(W, W.transpose(1, 2)) - torch.eye(W.size(1)).to(W.device).unsqueeze(0)
        return lambda_orthogonal * torch.norm(WT_W_minus_I, 'fro')

    def _regulaztion_loss_promptlen_2d(self, prompt_param):
        # reshape to 2D tensor：(prompt_len, prompt_dim*prompt_size*prompt_size)
        prompts_reshaped = prompt_param.view(prompt_param.size(1), -1)
        # Gram
        gram = torch.mm(prompts_reshaped, prompts_reshaped.t())
        # loss to eye
        identity = torch.eye(prompt_param.size(1)).to(prompt_param.device)
        loss = nn.MSELoss()(gram, identity) * self.lambda_orthogonal
        return loss
    
    def _regulaztion_loss_promptlen_cosine(self, prompt_param):
        '''orhogonal cosine'''
        # reshape to 2D tensor：(prompt_len, prompt_dim*prompt_size*prompt_size)
        prompts_reshaped = prompt_param.view(prompt_param.size(1), -1)
        prompts_normed = F.normalize(prompts_reshaped, dim=1)  # 归一化        
        # cosine
        cosine_sim_matrix = torch.mm(prompts_normed, prompts_normed.t())
        off_diagonal = cosine_sim_matrix - torch.eye(prompt_param.size(1)).to(prompt_param.device)
        # min eye
        loss = off_diagonal.abs().sum() * self.lambda_orthogonal
        return loss

    def _regulaztion_loss_promptlen_angle(self, prompt_param):
        '''
        intro:
            theta larger then the threshold. loss = max(0, cos(\theta_min) - AB / abs(a) * abs(b))
        '''
        loss = 0
        prompts = prompt_param
        cos_theta_min = 0 # equal to 90 degree
        for i in range(prompts.shape[1]):
            for j in range(i+1, prompts.shape[1]):
                vec_i = prompts[0, i].reshape(-1)
                vec_j = prompts[0, j].reshape(-1)
                cos_theta = torch.dot(vec_i, vec_j) / (torch.norm(vec_i) * torch.norm(vec_j))
                loss += torch.relu(cos_theta - cos_theta_min)
        return loss * self.lambda_orthogonal

    def get_detask_prompt_reg_loss(self, reg_opt, ):
        '''calculate the regularization loss, reg_opt to provide the choice'''
        # 
        assert reg_opt in ['3d', '2d', 'cosine', 'angle']
        if reg_opt == '3d':
            return self._regulaztion_loss_promptlen_3d(self.detask_prompt_param)
        elif reg_opt == '2d':
            return self._regulaztion_loss_promptlen_2d(self.detask_prompt_param)
        elif reg_opt == 'cosine':
            return self._regulaztion_loss_promptlen_cosine(self.detask_prompt_param)
        elif reg_opt == 'angle':
            return self._regulaztion_loss_promptlen_angle(self.detask_prompt_param)
        else:
            return torch.Tensor(0)

    # ------ Large Model processing --------
    @torch.no_grad()
    def _clip_preprocess(self,):
        device = "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_text_encoder = clip_model.encode_text
        text_token = clip.tokenize(self.task_text_prompt)
        clip_prompt = clip_text_encoder(text_token)
        return clip_prompt[:self.task_classes]
        # shape [task, 512]

    @torch.no_grad()
    def _init_SAM_Model(self, ):
        raise NotImplementedError('to large')
        sam_checkpoint = ".SAM_Checkpoint/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        import copy
        device = "cuda"
        sam_image_encoder = copy.deepcopy(sam.image_encoder)
        sam_image_encoder = sam_image_encoder.cuda()
        return sam_image_encoder

    @staticmethod
    def _preprocess_sam_image(x: torch.Tensor, img_size=1024) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        pixel_std = torch.tensor([58.395, 57.12, 57.375])
        # x = (x - pixel_mean) / pixel_std
        x = (x - pixel_mean.view(1, 3, 1, 1)) / pixel_std.view(1, 3, 1, 1)
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


    def _unify_sementic_prompt_shape(self, origin_prompt, opt = None):
        '''
        reshape degradation-aware-prompt, clip-prompt, SAM-prompt to the same dim
        prompt_shape:
            degradation-aware-prompt: (B, task len, dim, 1, 1)
            clip-prompt: (B, 512)
            SAM-prompt: (B, 256, 64, 64)
        '''
        prompt_size = self.prompt_size
         
        assert opt in ['task', 'clip', 'sam']
        if opt == 'task':
            return origin_prompt.mean(dim=1)

        elif opt == 'clip':
            B, C = origin_prompt.size(0), origin_prompt.size(1)
            p = origin_prompt.view(B, C, 1, 1)
            p = F.interpolate(p, size=(prompt_size, prompt_size))
            return self.clip_conv(p)
    
        elif opt == 'sam':
            p =  self.sam_conv(origin_prompt)
            p = F.interpolate(p, size=(prompt_size, prompt_size))
            return p

    def _angle_to_cosine(angle_degrees):
        angle_radians = math.radians(angle_degrees)
        cosine_value = math.cos(angle_radians)
        return cosine_value






# PIP MODULE 
class PromptInPrompt(Basic_prompt):
    '''
    '''
    def __init__(self, prompt_dim = 64, 
                task_classes = 5, 
                prompt_size = 96, 
                lin_dim = 192,
                use_detask_label = True,
                use_degradation_sensor = False,
                use_detask_prompt = True,
                use_SAM_prompt = False,
                use_CLIP_prompt = False,
                low_prompt_dim = 64
                ):
        super(PromptInPrompt,self).__init__()
        
        # all options
        self.use_degradation_sensor = use_degradation_sensor
        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 
                        'derain': 3, 'dehaze': 4, 
                        'deblur' : 5, 'enhance': 6}
        self.task_dict = {0: 'denoise_15', 1: 'denoise_25', 2: 'denoise_50', 3: 'derain', 4: 'dehaze', 5: 'deblur', 6: 'enhance'}
        self.task_text_prompt = [
            "Image denoising task with a noise level of sigma=15.",
            "Image denoising task with a noise level of sigma=25.",
            "Image denoising task with a noise level of sigma=50.",
            "Task of removing rain from an image, known as deraining.",
            "Image dehazing task to clear out the haze and improve visibility.",
            "Image deblurring task aimed at sharpening blurry images to retrieve details.",
            "Low light image enhancement task focused on amplifying and improving details in dimly lit images.",
        ]

        # 【1】 prompt type
        self.use_detask_prompt, self.use_SAM_prompt, self.use_CLIP_prompt = use_detask_prompt, use_SAM_prompt, use_CLIP_prompt
        self.use_detask_label = use_detask_label
        self.task_classes = task_classes
        self.prompt_size = prompt_size
        self.low_prompt_dim = low_prompt_dim

        # 【1.1】degradation-aware prompt
        self.detask_prompt_param = nn.Parameter(torch.randn(1, task_classes, low_prompt_dim, 1, 1)) 
        # 【1.2】 basic restoration prompt
        self.low_prompt_param = nn.Parameter(torch.randn(1,low_prompt_dim,prompt_size,prompt_size))

        # control outside is better
        if use_degradation_sensor:
            raise NotImplementedError('will update the sensor soon')
            self.DSM = None
            self.task_liner = nn.Linear(lin_dim, task_classes)
        else:
            pass

        # 【2】 SAM and CLIP prompts -- provide sementic prompt
        if self.use_SAM_prompt:
            self.sam_conv = nn.Conv2d(265, prompt_dim, 1, 1, 0)
            self.sam_image_encoder = self._init_SAM_Model()
        if self.use_CLIP_prompt:
            self.clip_conv = nn.Conv2d(512, prompt_dim , 1, 1, 0)
            self.clip_prompt = self._clip_preprocess()
        
        # 【4. universal restoration prompt by prompt in prompt learning】
        # prompt interaction
        prompt_transformer = CrossTransformerRestormer_BlockV2(dim = low_prompt_dim, num_heads = 2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
        self.LGM = prompt_transformer

        # 【5. output layer】 reduce channel from 2*dim -> 1*dim
        self.outconv = nn.Conv2d(low_prompt_dim * 1, 
                                 low_prompt_dim, 
                                 kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x, degradation_class = None):
        B,C,H,W = x.shape        
        # 【1】 degradation task prompt
        if self.use_detask_label:
            if degradation_class is None:
                raise ValueError("during training, should provide degradation clss if use_detask_label")
            
            if isinstance(degradation_class, list):
                mixed_one_hot_labels = torch.stack([F.one_hot(pair[0], num_classes=self.task_classes) + F.one_hot(pair[1], num_classes=self.task_classes) 
                                    for pair in degradation_class]) / 2
                detask_prompt_weights = mixed_one_hot_labels
            else: 
                detask_prompt_weights = torch.nn.functional.one_hot(degradation_class, num_classes = self.task_classes).to(x.device) # .cuda() 
        else:
            raise NotImplementedError
            # degradation aware model has been moved outside
            detask_prompt_weights = self.DSM(x) 

        if self.use_detask_prompt:
            # (B, prompt_len, 1, 1, 1) * (B, prompt_len, prompt_dim, size, size)
            detask_prompts = detask_prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.detask_prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
            detask_prompt = torch.mean(detask_prompts, dim = 1)
            # (B, dim, size, size)

        # 【1.1】 other high prompt with CLIP and SAM
        if self.use_CLIP_prompt:
            # with torch.no_grad():
            # clip_prompt = self.clip_prompt[:self.].clone() # (B, 512)
            clip_prompt = self.clip_prompt.detach().to(x.device)
            clip_prompt = detask_prompt_weights.unsqueeze(-1) * clip_prompt.unsqueeze(0).repeat(B,1,1)
            clip_prompt = torch.mean(clip_prompt, dim = 1) # (B, 512)
            clip_prompt = self._unify_sementic_prompt_shape(clip_prompt, opt='clip') # (B, dim, s, s)
        if self.use_SAM_prompt:
            with torch.no_grad():
                raise NotImplementedError('not supported SAM, too large') 
                print(x.shape)
                sam_input = self._preprocess_sam_image(x, img_size = 1024)
                sam_prompt = self.sam_image_encoder(sam_input) # (B, 256, 64, 64)
                sam_prompt = self._unify_sementic_prompt_shape(sam_prompt, opt='sam')

        # 【2】combine all the prompt
        selected_sementic_prompts = []
        if self.use_detask_prompt: 
            selected_sementic_prompts.append(detask_prompt)
        if self.use_CLIP_prompt:
            selected_sementic_prompts.append(clip_prompt)
        if self.use_SAM_prompt:
            raise NotImplementedError('not supported SAM, too large')
            selected_sementic_prompts.append(sam_prompt)

        if selected_sementic_prompts:
            sementic_prompt = torch.cat(selected_sementic_prompts, dim = 1)
            feature = F.interpolate(x,(self.prompt_size,self.prompt_size),mode="bilinear")
            # algin the feature channel size to prompt
            if C > self.low_prompt_dim:
                feature = feature[:,:self.low_prompt_dim,:,:]
            elif C < self.low_prompt_dim:
                padding_size = self.low_prompt_dim - C
                zero_padding = torch.zeros(feature.size(0), padding_size, *feature.shape[2:], requires_grad=True).to(feature.device)
                feature = torch.cat([feature, zero_padding], dim=1)
            else: # equal
                pass
            sementic_prompt = sementic_prompt * feature

            low_prompt = self.LGM(self.low_prompt_param.repeat(B, 1, 1, 1), sementic_prompt)
        else:
            raise ValueError("NOT THE CURRENT VERSION")
            low_prompt = self.low_prompt_param.repeat(B, 1, 1, 1)

        # 【3】 output
        output_prompt = F.interpolate(low_prompt,(H,W),mode="bilinear")
        output_prompt = self.outconv(output_prompt)
        if self.use_detask_label:
            return output_prompt, detask_prompt_weights
        else:
            return output_prompt, detask_prompt_weights



class PromptToFeature(nn.Module):
    '''
    prompt-to-feature in PIP
    '''
    def __init__(self, 
                interaction_mode = 'promptir',
                interaction_opt = {},
                ):
        super(PromptToFeature, self).__init__()

        self.interaction_mode, self.interaction_opt = interaction_mode, interaction_opt

        if interaction_mode == "promptir":
            params = ['feat_dim', 'prompt_dim', 'head', 'ffn_expansion_factor', 'LayerNorm_type', 'bias']
            feat_dim, prompt_dim, head, ffn_expansion_factor, LayerNorm_type, bias = (interaction_opt[param] for param in params)
            self.prompt_transformer = TransformerBlock_Restormer(dim=int(feat_dim) + prompt_dim, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            self.reduce_channel = nn.Conv2d(int(feat_dim)+prompt_dim,int(feat_dim),kernel_size=1,bias=bias)
        
        elif interaction_mode == "convgate":
            in_feat, out_feat = interaction_opt["in_feat"], interaction_opt["out_feat"]
            kernel_size = 3
            bias=False 
            self.conv1 = nn.Conv2d(in_feat, in_feat, kernel_size, bias=bias, padding = 1)
            self.conv2 = nn.Conv2d(in_feat, out_feat, kernel_size, bias=bias, padding = 1)
            self.conv3 = nn.Conv2d(in_feat, out_feat, kernel_size, bias=bias, padding = 1)

        elif interaction_mode == 'promptir_gate':
            params = ['feat_dim', 'prompt_dim', 'head', 'ffn_expansion_factor', 'LayerNorm_type', 'bias']
            feat_dim, prompt_dim, head, ffn_expansion_factor, LayerNorm_type, bias = (interaction_opt[param] for param in params)
            self.prompt_transformer = TransformerBlock_Restormer(dim=int(feat_dim) + prompt_dim, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            self.gate_reduce_channel = nn.Conv2d(int(feat_dim)+prompt_dim, int(feat_dim), kernel_size=3, bias=bias,padding=1) 
            self.noise_channel_reduce = nn.Conv2d(int(feat_dim)+prompt_dim,int(feat_dim),kernel_size=1,bias=bias,padding=0) 
        
        elif interaction_mode == 'pip_cross':
            params = ['feat_dim', 'prompt_dim', 'head', 'ffn_expansion_factor', 'LayerNorm_type', 'bias']
            feat_dim, prompt_dim, head, ffn_expansion_factor, LayerNorm_type, bias = (interaction_opt[param] for param in params)
            self.prompt_transformer_cross = CrossTransformerRestormer_Block_PIM(feat_dim, prompt_dim, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        
        elif interaction_mode == 'pip_cross_topm':
            params = ['feat_dim', 'prompt_dim', 'head', 'ffn_expansion_factor', 'LayerNorm_type', 'bias']
            feat_dim, prompt_dim, head, ffn_expansion_factor, LayerNorm_type, bias = (interaction_opt[param] for param in params)
            self.prompt_transformer_cross = Topm_CrossTransformerRestormer_Block_PIM(feat_dim, prompt_dim, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        else:
            raise NotImplementedError

    def forward(self, x, prompt):
        if self.interaction_mode == 'promptir':
            y = self._forward_promptir(x, prompt)
        elif self.interaction_mode == 'convgate':
            y = self._forward_my_convgate(x, prompt)
        elif self.interaction_mode == 'promptir_gate':
            y = self._forward_promptir_gate(x, prompt)
        elif self.interaction_mode == 'pip_cross':
            y = self._forward_pip_cross(x, prompt)
        elif self.interaction_mode == 'pip_cross_topm':
            y = self._forward_pip_topm_cross(x, prompt)
        else:
            raise NotImplementedError("unspported prompt to feature interaction mode")
        return y
    
    def _forward_promptir(self, x, prompt):
        x = torch.cat([x, prompt], dim = 1)
        x = self.prompt_transformer(x, prompt)
        y = self.reduce_channel(x)
        return y

    def _forward_my_convgate(self, x, prompt):
        x_mid = self.conv1(prompt)
        gate = torch.sigmoid(self.conv2(x_mid))
        noise = self.conv3(x_mid)
        y = x * gate + noise
        return  y

    def _forward_promptir_gate(self, x, prompt):
        x_in = torch.cat([x, prompt], dim = 1)
        x_mid = self.prompt_transformer(x_in)
        gate = torch.sigmoid(self.gate_reduce_channel(x_mid))
        noise = self.noise_channel_reduce(x_mid)
        y = x * gate  + noise
        return  y

    def _forward_pip_cross(self, x, prompt):
        y = self.prompt_transformer_cross(x, prompt)
        return y

    def _forward_pip_topm_cross(self, x, prompt):
        y = self.prompt_transformer_cross(x, prompt)
        return y

    def _forward_convgate_paper(self, x, prompt):
        raise NotImplementedError("bad")

    def _forward_convgate_origin(self, x, prompt):
        raise NotImplementedError("bad")













if __name__ == '__main__':
    pass














