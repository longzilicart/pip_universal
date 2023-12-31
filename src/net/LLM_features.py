# get LLM features from SAM Clip and so on


import torch
import torch.nn as nn
from torch.nn import functional as F
# other encoder
# import sys
# sys.path.append("../..") # to the basic folder, adding sam
# from segment_anything.segment_anything import sam_model_registry
import copy
import torch
import clip 
# pretrained clip text/image encoder
# ! pip install ftfy regex tqdm
# ! pip install git+https://github.com/openai/CLIP.git




class SAM_Image_Encoder(nn.Module):
    '''get pretrained SAM image encoder
    
    '''
    def __init__(self, sam_checkpoint, model_type = 'vit_b'):
        sam_checkpoint = "/mnt/Tempdrive2/workspace/Universal_Restoration/pretrain_models/SAM_Checkpoint/sam_vit_b_01ec64.pth"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) #, image_size = 512)
        self.sam_image_encoder = copy.deepcopy(sam.image_encoder)
    
    @staticmethod
    def preprocess_sam(x: torch.Tensor, img_size=1024) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        
        # do we really need that
        pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        pixel_std = torch.tensor([58.395, 57.12, 57.375])
        x = (x - pixel_mean.view(1, 3, 1, 1)) / pixel_std.view(1, 3, 1, 1)
        # Pad 填充到self.image_encoder.img_size的正方形 # SAM已经规定了1024 # 有点疑问
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        y = F.pad(x, (0, padw, 0, padh))
        return y

    @torch.no_grad()
    def forward(self, x):
        '''
        get features from SAM image encoder
        output shape: (B, 512, 64, 64)
        '''
        x = self.preprocess_sam(x)
        y = self.sam_image_encoder(x)
        return y



class CLIP_Text_Encoder(nn.Module):
    '''get pretrained CLIP text encoder
    '''
    def __init__(self, clip_checkpoint, model_type = ''):

        # self.clip_text_encoder = 
        # TODO make sure everything is right here
        clip_model, preprocess = clip.load("ViT-B/32", device='cpu')
        self.clip_text_encoder = clip_model.encode_text

    @torch.no_grad()
    def forward(self, text):
        '''
        get text embedding from CLIP
        '''
        # text shoule be in [B, length]
        text_token = clip.tokenize(text)
        text_features = self.clip_text_encoder(text_token)
        return text_features
    

class CLIP_Image_Encoder(nn.Module):
    '''get pretrained CLIP image encoder
    '''
    def __init__(self, clip_checkpoint, model_type = ''):

        # TODO make sure everything is right here
        clip_model, preprocess = clip.load("ViT-B/32", device='cpu')
        self.clip_image_encoder = clip_model.encode_image

    @torch.no_grad()
    def forward(self, image):
        pre_image = self.preprocess_clip_image(image, img_size=224)
        clip_image = self.clip_image_encoder(pre_image)
        # print(clip_image.shape)
        print(clip_image)

    @staticmethod
    def preprocess_clip():
        pass




if __name__ == '__main__':
    pass






