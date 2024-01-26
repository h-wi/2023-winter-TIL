import torch
import math
import torch.nn as nn
from timm import create_model
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F
from timm.models.registry import register_model

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor

    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

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

        self.attn_drop_p = attention_dropout

    def forward(self, x):
        B, N, C = x.shape # 128,256,256

        # x = self.qkv(x).reshape(3, B, self.num_heads, N, self.head_dim)
        x = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = (x[0], x[1], x[2])

        # if True:
        #     x = F._scaled_dot_product_attention(
        #         query,
        #         key,
        #         value,
        #         dropout_p=self.attn_drop_p if self.training else 0.,
        #     )
        # else:
        attn = torch.matmul(query, key.transpose(2,3)) * self.scale # (128, 4, 256, 256) = (batch_size, head_num, patch_num, dim)
        attn = attn.softmax(dim=-1) # 한 패치에 대한 dim의 attention을 softmax
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, value) # (128, 4, 256, 64)

        # if scaled_dot_product_attn ~
        # print(x[0].size()) # Tensor, (128,4,256,64)
        # print(type(x[0]))
        # print(x[1].size()) # Tensor, (128,4,256,256)
        # print(type(x[1]))

        x = x.transpose(1,2) # (128, 256, 4, 64)
        x = x.reshape(B,N,C) # (128, 256, 256)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# window attention
class W_MSA(nn.Module):
    def __init__(self,
                 dim, num_heads, head_dim=None, window_size=8,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # setting
        B, N, C = x.shape # [128, 256, 256]
        window = self.window_size # 8
        w = h = int(math.sqrt(N)) # 16x16이면 4x4가 하나의 토큰이 되게 w = h = 16
        h_ = int(h // window) # 여기에 window attention 가능하게 window 크기만큼 나눔. # 16 // 8 = 2
        w_ = int(w // window) # 2

        x = x.view(B, h, w, C)  # [B, H, W, C] = [128, 16, 16, 256]
        x = x.view(B, h_, window, w_, window, C)  # [0, 1, 2, 3, 4, 5 ] -> [0, 1, 3, 2, 4, 5 ] - change index
                                                  # [128, 2, 8, 2, 8, 256]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * h_ * w_, window * window, C)  # [B*2*2, 8*8, 256]

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.view(B, h_, w_, window, window, C)              # [B, 8, 8, 7, 7, 96]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, 8, 7, 8, 7, 96]
        x = x.view(B, h, w, -1)               # (roll)  [B, 56, 56, 96]
        x = x.view(B, h * w, C)                       # [B, 56, 56, 96]
        return x

# we only select the top-k similar tokens from the keys for each query to compute the attention map.
# k-NN Attention ViT = KVT

class KNNAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0., topk=100):
        super().__init__()
        assert dim % num_heads == 0
        self.topk = topk # 100개?
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.zeros(B, self.num_heads, N, N).to('cuda:0')

        index = torch.topk(attn, k=self.topk, dim=-1)[1] # Channel 256개 중 100개에만 연산, 나머지는 masking
        # print(index.shape)

        mask.scatter_(-1, index, 1).to()
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float("-inf")))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.activation = F.gelu

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src):
        src = self.linear1(src)
        src = self.activation(src)
        src = self.dropout1(src)
        src = self.linear2(src)
        src = self.dropout2(src)

        return src

class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # self.attn = Attention(dim=d_model, num_heads=nhead,
        #                       attention_dropout=attention_dropout,
        #                       projection_dropout=dropout)
        # self.attn = W_MSA(dim=d_model, num_heads=nhead,
        #                   attn_drop=attention_dropout,
        #                   proj_drop=dropout)
        self.attn = KNNAttention(dim=d_model, num_heads=nhead,
                                 attn_drop=attention_dropout,
                                 proj_drop=dropout)
        self.mlp = MLP(d_model, dim_feedforward, dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()


    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.attn(self.norm1(src)))
        src = src + self.drop_path(self.mlp(self.norm2(src)))
        return src


class TransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

class CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def _cct(arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         positional_embedding='learnable',
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CCT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                *args, **kwargs)

    return model


def cct_7(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


@register_model
def cct_7_3x1_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_7('cct_7_3x1_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


if __name__ == '__main__':
    x = torch.rand(2, 3, 32, 32)
    model = create_model('cct_7_3x1_32', num_classes=100)
    y = model(x)
    print(y.shape)