
import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import math

# other LLM 
import clip
import sys
sys.path.append("../..")
# from segment_anything.segment_anything import sam_model_registry




###########################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



###################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward_Restormer(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_Restormer, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention_Restormer (MDTA)
class Attention_Restormer(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_Restormer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class resblock_restormer(nn.Module):
    def __init__(self, dim):
        super(resblock_restormer, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree') # TODO 这个是不需要？
        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



##########################################################################
## Transformer Block
class TransformerBlock_Restormer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_Restormer, self).__init__()
        # self.dim = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_Restormer(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_Restormer(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



###### Dual Gated Feed-Forward Networ from AAAI23 paper
class FeedForward_DualGate(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_DualGate, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2)*x1 + F.gelu(x1)*x2
        x = self.project_out(x)
        return x











##########################################################################
##########################################################################
# basic module in pip
class CrossAttention_RestormerV2(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention_RestormerV2, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x_q, x_kv):
        '''
        input:
            x_q: for query
            x_kv: for key and value
        normally:
            feature is x_q, and prompt is x_kv
            this imple that feature will select the prompt
        cross attention as no skip connection in default
        '''
        b,c,h,w = x_q.shape
        q = self.q_dwconv(self.q(x_q))
        kv = self.kv_dwconv(self.kv(x_kv))
        k,v = kv.chunk(2, dim=1)  
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
    

## Transformer Block
class CrossTransformerRestormer_BlockV2(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                cross_residual = True):
        super(CrossTransformerRestormer_BlockV2, self).__init__()
        # self.dim = dim
        self.norm11 = LayerNorm(dim, LayerNorm_type)
        self.norm12 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention_RestormerV2(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_Restormer(dim, ffn_expansion_factor, bias)
        # other config
        self.cross_residual = cross_residual

    def forward(self, x_q, x_kv):
        if self.cross_residual:
            x_attn = x_q + self.attn(self.norm11(x_q), self.norm12(x_kv)) 
        else:
            x_attn = self.attn(self.norm11(x_q), self.norm12(x_kv)) 
            
        y = x_attn + self.ffn(self.norm2(x_attn))
        return y





# ================ some useful tools ==============


###### Dual Gated Feed-Forward Network
class DualGate_FeedForward(nn.Module):
    '''from LLformer (https://github.com/TaoWangzj/LLFormer), which is modified from restormer'''
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DualGate_FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2)*x1 + F.gelu(x1)*x2
        x = self.project_out(x)
        return x


###### Mixed-Scale Feed-forward Network (MSFN)
class MixScale_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(MixScale_FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)
        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)
        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))
        x = torch.cat([x1, x2], dim=1)
        x = self.project_out(x)
        return x


# prompt to feature interaction module in PIP
class CrossTransformerRestormer_Block_PIM(nn.Module):
    '''
    cross attention in prompt-to-feature interaction
    intro:
        It supports features and prompt with different channels. The prompt channel is clipped and aligned to the feature map. 
    '''
    def __init__(self, feat_dim, prompt_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                cross_residual = True):
        super(CrossTransformerRestormer_Block_PIM, self).__init__()
        self.norm11 = LayerNorm(feat_dim, LayerNorm_type)
        self.norm12 = LayerNorm(prompt_dim, LayerNorm_type)
        if feat_dim >= prompt_dim:
            self.attn = CrossAttention_RestormerV2(prompt_dim, num_heads, bias)
        else: # topm attention
            self.attn = CrossAttention_RestormerV2(prompt_dim, num_heads, bias)
            self.reduce_channel = nn.Conv2d(prompt_dim, int(feat_dim),kernel_size=1,bias=False)

        # based on feature dimension
        self.norm2 = LayerNorm(feat_dim, LayerNorm_type)
        self.ffn = DualGate_FeedForward(feat_dim, ffn_expansion_factor, bias)
        # other config 
        self.cross_residual = cross_residual

    def forward(self, x_q, x_kv):
        x_q, x_kv = self.norm11(x_q), self.norm12(x_kv)
        q_c, kv_c = x_q.size(1), x_kv.size(1)
        if q_c > kv_c:
            x_q_inter, x_q_stable = torch.split(x_q, [kv_c, q_c-kv_c], dim=1)
            x_kv_inter = x_kv
        elif q_c < kv_c:
            padding_size = kv_c - q_c
            # zero padding on channel when q < kv
            zero_padding = torch.zeros(x_q.size(0), padding_size, *x_q.shape[2:], requires_grad=True).to(x_q.device)
            x_q_inter = torch.cat([x_q, zero_padding], dim=1)
            x_q_stable = None
            x_kv_inter = x_kv
        else:
            x_q_inter, x_q_stable, x_kv_inter = x_q, None, x_kv

        if self.cross_residual:
            x_attn = x_q_inter + self.attn(x_q_inter, x_kv_inter) 
        else:
            x_attn = self.attn(x_q_inter, x_kv_inter) 

        if q_c > kv_c:  
            x_attn = torch.cat([x_attn, x_q_stable], dim=1)
        elif q_c < kv_c:
            x_attn = self.reduce_channel(x_attn)

        y = x_attn + self.ffn(self.norm2(x_attn))
        return y




##  Top-m Sparse Attention (TKSA)
class Topm_CrossAttention_Restormer(nn.Module):
    '''
    top-m cross attention module
    intro:
        modified from paper https://github.com/cschenxiang/DRSformer
    '''
    def __init__(self, dim, num_heads, bias):
        super(Topm_CrossAttention_Restormer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x_q, x_kv):
        b,c,h,w = x_q.shape
        q = self.q_dwconv(self.q(x_q))
        kv = self.kv_dwconv(self.kv(x_kv))
        k,v = kv.chunk(2, dim=1)  
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class Topm_CrossTransformerRestormer_Block_PIM(nn.Module):
    '''
    top-m cross attention in prompt-to-feature interaction
    intro:
        It supports features and prompt with different channels. The prompt channel is clipped and aligned to the feature map. 
    '''
    def __init__(self, feat_dim, prompt_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                cross_residual = True):
        super(Topm_CrossTransformerRestormer_Block_PIM, self).__init__()
        self.norm11 = LayerNorm(feat_dim, LayerNorm_type)
        self.norm12 = LayerNorm(prompt_dim, LayerNorm_type)
        if feat_dim >= prompt_dim:
            self.attn = Topm_CrossAttention_Restormer(prompt_dim, num_heads, bias)
        else:
            self.attn = Topm_CrossAttention_Restormer(prompt_dim, num_heads, bias)
            self.reduce_channel = nn.Conv2d(prompt_dim, int(feat_dim),kernel_size=1,bias=False)

        self.norm2 = LayerNorm(feat_dim, LayerNorm_type)
        self.ffn = DualGate_FeedForward(feat_dim, ffn_expansion_factor, bias)
        # other config
        self.cross_residual = cross_residual

    def forward(self, x_q, x_kv):
        '''
        prompt interaction中, feature应该作为q, prompt作为kv
        '''
        x_q, x_kv = self.norm11(x_q), self.norm12(x_kv)
        q_c, kv_c = x_q.size(1), x_kv.size(1)
        if q_c > kv_c:
            x_q_inter, x_q_stable = torch.split(x_q, [kv_c, q_c-kv_c], dim=1)
            x_kv_inter = x_kv
        elif q_c < kv_c:
            padding_size = kv_c - q_c
            zero_padding = torch.zeros(x_q.size(0), padding_size, *x_q.shape[2:], requires_grad=True).to(x_q.device)
            x_q_inter = torch.cat([x_q, zero_padding], dim=1)
            x_q_stable = None
            x_kv_inter = x_kv
        else:
            x_q_inter, x_q_stable, x_kv_inter = x_q, None, x_kv

        if self.cross_residual:
            x_attn = x_q_inter + self.attn(x_q_inter, x_kv_inter) 
        else:
            x_attn = self.attn(x_q_inter, x_kv_inter) 

        if q_c > kv_c:  
            x_attn = torch.cat([x_attn, x_q_stable], dim=1)
        elif q_c < kv_c:
            x_attn = self.reduce_channel(x_attn)

        y = x_attn + self.ffn(self.norm2(x_attn))
        return y
