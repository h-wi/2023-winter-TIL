import torch
import torch.nn as nn
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init

class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5 # 제곱

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.proj = Linear(dim, dim)

        self.attn_drop = Dropout(attention_dropout)
        self.proj_drop = Dropout(projection_dropout)

    def use_fused_attn(self):
        # TODO: check whether it is scaled dot product attention
        return False

# TODO : Attention forward 구현하기
    def forward(self, x):
        B, N, C = x.shape # 128,256,256

        # x = self.qkv(x).reshape(3, B, self.num_heads, N, self.head_dim)
        x = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = (x[0], x[1], x[2])

        if self.use_fused_attn():
            print('check')
        else:
            attn = torch.matmul(query, key.transpose(2,3)) * self.scale # (128, 4, 256, 256) = (batch_size, head_num, patch_num, dim)
            attn = attn.softmax(dim=-1) # 한 패치에 대한 dim의 attention을 softmax
            attn = self.attn_drop(attn)
            x = torch.matmul(attn, value) # (128, 4, 256, 64)

        x = x.transpose(1,2) # (128, 256, 4, 64)
        x = x.reshape(B,N,C) # (128, 256, 256)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

