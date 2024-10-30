import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Reduce
import torch.nn.functional as F

# helpers

def exists(val):
    return val is not None

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv3d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv3d(dim * mult, dim, 1)
    )

# MHRAs (multi-head relation aggregators)

class LocalMHRA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        local_aggr_kernel = 5
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        # they use batchnorm for the local MHRA instead of layer norm
        self.norm = nn.BatchNorm3d(dim)

        # only values, as the attention matrix is taking care of by a convolution
        self.to_v = nn.Conv3d(dim, inner_dim, 1, bias = False)

        # this should be equivalent to aggregating by an attention matrix parameterized as a function of the relative positions across each axis
        self.rel_pos = nn.Conv3d(heads, heads, local_aggr_kernel, padding = local_aggr_kernel // 2, groups = heads)

        # combine out across all the heads
        self.to_out = nn.Conv3d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        b, c, *_, h = *x.shape, self.heads

        # to values
        v = self.to_v(x)

        # split out heads
        v = rearrange(v, 'b (c h) ... -> (b c) h ...', h = h)

        # aggregate by relative positions
        out = self.rel_pos(v)

        # combine heads
        out = rearrange(out, '(b c) h ... -> b (c h) ...', b = b)
        return self.to_out(out)

class GlobalMHRA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        shape, h = x.shape, self.heads

        x = rearrange(x, 'b c ... -> b c (...)')

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n', h = h)

        out = self.to_out(out)
        return out.view(*shape)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        mhsa_type = 'g',
        local_aggr_kernel = 5,
        dim_head = 64,
        ff_mult = 4,
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mhsa_type == 'l':
                attn = LocalMHRA(dim, heads = heads, dim_head = dim_head, local_aggr_kernel = local_aggr_kernel)
            elif mhsa_type == 'g':
                attn = GlobalMHRA(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv3d(dim, dim, 3, padding = 1),
                attn,
                FeedForward(dim, mult = ff_mult, dropout = ff_dropout),
            ]))

    def forward(self, x):
        for dpe, attn, ff in self.layers:
            x = dpe(x) + x
            x = attn(x) + x
            x = ff(x) + x

        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class UpRegressor(nn.Module):
    def __init__(self, dim=512):
        super(UpRegressor, self).__init__()
        self.spat = nn.AdaptiveAvgPool3d((None, 2, 2))
        # k = 2p + s
        self.temp = nn.Sequential(
            # nn.ConvTranspose3d(in_channels=dim, out_channels=dim, kernel_size=(4,1,1), padding=(1,0,0), stride=(2,1,1)),
            nn.Conv3d(in_channels=dim, out_channels=dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(dim // 2),
            QuickGELU()
        )
        self.regress = nn.Conv3d(in_channels=dim // 2, out_channels=1, kernel_size=1)
    
    def forward(self, x):
        # B C T H W
        if len(x.shape) == 5:  # avg pool
            x = self.spat(x)
        x = F.interpolate(x, scale_factor=(2, 1, 1)) 
        x = self.temp(x)  # b c t
        return self.regress(x)


class Uniformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dims = (64, 64, 64, 64),
        depths = (2, 2, 2, 2),
        mhsa_types = ('l', 'l', 'g', 'g'),
        local_aggr_kernel = 5,
        channels = 3,
        ff_mult = 4,
        dim_head = 64,
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()
        self.S = 2
        init_dim, *_, last_dim = dims
        self.to_tokens = nn.Conv3d(channels, init_dim, (3, 4, 4), stride = (2, 4, 4), padding = (1, 0, 0))

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))
        mhsa_types = tuple(map(lambda t: t.lower(), mhsa_types))

        self.stages = nn.ModuleList([])

        for ind, (depth, mhsa_type) in enumerate(zip(depths, mhsa_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Transformer(
                    dim = stage_dim,
                    depth = depth,
                    heads = heads,
                    mhsa_type = mhsa_type,
                    ff_mult = ff_mult,
                    ff_dropout = ff_dropout,
                    attn_dropout = attn_dropout
                ),
                nn.Sequential(
                    nn.Conv3d(stage_dim, dims[ind + 1], (1, 2, 2), stride = (1, 2, 2)),
                    LayerNorm(dims[ind + 1]),
                ) if not is_last else None
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b c t h w -> b c', 'mean'),
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, num_classes)
        )
        self.regress = UpRegressor(last_dim)

    def forward(self, video):
        
        means = torch.mean(video, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(video, dim=(2, 3, 4), keepdim=True)
        video = (video - means) / stds
        
        # print("original: ", video.shape)
        x = self.to_tokens(video)
        # print("after to_tokens: ", x.shape)

        for transformer, conv in self.stages:
            x = transformer(x)
            # print("transformer: ", x.shape)

            if exists(conv):
                x = conv(x)
                # print("conv: ", x.shape)
        
        x = self.regress(x)
        # print("regress: ", x.shape)
        
        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:,:,:,a,b]) # (B, 1, T)
        # print("---------len(x_list): ", len(x_list)) # len(x_list): 4
        # print("---------x.shape (in list): ", x_list[0].shape) # torch.Size([2, 1, 300])
        x = sum(x_list)/(self.S*self.S) # (B, 1, T) 
        # print("---------x.shape (after normalize): ", x.shape) # torch.Size([2, 1, 300])
        X = torch.cat(x_list+[x], 1) # (B, M, T), flatten all spatial signals to the second dimension
        # print("---------X.shape: ", X.shape) # torch.Size([2, 5, 300])
        return X
    
if __name__ == '__main__':
    model = Uniformer(
        num_classes = 1000,                 # number of output classes
        dims = (64, 64, 64, 64),
        depths = (2, 2, 2, 2),
        mhsa_types = ('l', 'l', 'g', 'g')   # aggregation type at each stage, 'l' stands for local, 'g' stands for global
    )

    video = torch.randn(2, 3, 300, 128, 128)  # (batch, channels, time, height, width)
    print("output: ", model(video).shape)
