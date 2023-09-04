
"""
INPUT: difference feature map dF(b,256,x,y) from stags 1 --> 4

Perform token and channel mixing on individual dFs
At the end, spatially scale the dFs to match image res.

OUTPUT: enriched dF (b,256,256,256)

reference: github.com/lucidrains/mlp-mixer-pytorch
"""

from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
    #token-mix, chann-mix

    """
    Steps:
        1. Rearrange image to 256 non-overlapping linear patches of length 512.
        2. Reduce channels from 768 -> 512
        3. Perform MLP1 (token mixing)
        4. Perform MLP2 (channel mixing)
    """

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim), #(768,512)
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)), #MLP1
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))     #MLP2
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = 16, p1 = patch_size, p2 = patch_size)
        # Reduce('b n c -> b c', 'mean'),
        # nn.Linear(dim, num_classes)
    )