import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from models.encoder import EncoderBlock, CEFF, LayerNorm
from models.mixer import MLPMixer

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # input x (8,3,256,256)
        x = self.proj(x) # (8,64,64,64)
        _, _, H, W = x.shape
        #x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)  # stage1 (8,64,64,64)

        return x, H, W

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)

            if output_h > input_h or output_w > output_h:

                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):

                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# Transformer Decoder MLP
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


#Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

#Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # in 256, out 256
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


#Transormer Ecoder with x2, x4, x8, x16 scales
class EncoderTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256],
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=LayerNorm, depths=[3, 3, 6, 18]):
        super().__init__()
        self.num_classes    = num_classes
        self.depths         = depths
        self.embed_dims     = embed_dims   # (64,128,320,512)

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0 

        self.block1 = nn.ModuleList([EncoderBlock(dim=embed_dims[0], dim_head=4)
                                     for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        
        # Stage-2 (x1/8 scale)
        cur += depths[0]

        self.block2 = nn.ModuleList([EncoderBlock(dim=embed_dims[1], dim_head=4)
                                     for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
       
       # Stage-3 (x1/16 scale)
        cur += depths[1]
        
        self.block3 = nn.ModuleList([EncoderBlock(dim=embed_dims[2], dim_head=8)
                                     for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        
        # Stage-4 (x1/32 scale)
        cur += depths[2]

        self.block4 = nn.ModuleList([EncoderBlock(dim=embed_dims[3], dim_head=8)
                                     for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_features(self, feats):
        B = feats.shape[0]  # batchsize
        outs = []
    
        # stage 1
        feats, H1, W1 = self.patch_embed1(feats)
        for i, blk in enumerate(self.block1):
            feats = blk(feats, H1, W1) # (8,64,64,64)
        feats = self.norm1(feats)
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)

        #stage 2
        feats, H1, W1 = self.patch_embed2(feats)
        for i, blk in enumerate(self.block2):
            feats = blk(feats, H1, W1)
        feats = self.norm2(feats)
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)

        # stage 3
        feats, H1, W1 = self.patch_embed3(feats)
        for i, blk in enumerate(self.block3):
            feats = blk(feats, H1, W1)
        feats = self.norm3(feats)
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)

        # stage 4
        feats, H1, W1 = self.patch_embed4(feats)
        for i, blk in enumerate(self.block4):
            feats = blk(feats, H1, W1)
        feats = self.norm4(feats)
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

class DecoderTransformer(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, align_corners=True, in_channels=[64, 128, 320, 512], embedding_dim=256, output_nc=2, decoder_softmax=False):
        super(DecoderTransformer, self).__init__()
        
        #settings
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim  # output.dim = input.dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        
        #Final linear fusion layer

        #desired_in_channels = int(embedding_dim*len(in_channels)/2)
        desired_in_channels = int(embedding_dim*len(in_channels))
        self.linear_fuse = nn.Sequential(
           nn.Conv2d(in_channels=desired_in_channels, out_channels=self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
      
        self.ceff1 = CEFF(in_channels=self.embedding_dim, height=2)
        self.ceff2 = CEFF(in_channels=self.embedding_dim, height=2)
        self.ceff3 = CEFF(in_channels=self.embedding_dim, height=2)
        self.ceff4 = CEFF(in_channels=self.embedding_dim, height=2)

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid()

    def forward(self, x_1, x_2):

        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1  # C1 (8,64,64,64), C2 (8,128,32,32)
        c1_2, c2_2, c3_2, c4_2 = x_2  # C3 (8,320,16,16), C4 (8,512,8,8)


        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3]) #(8,256,8,8)
        _c4   = self.ceff1([_c4_1, _c4_2])  # (8,256,8,8)
        p_c4  = self.make_pred_c4(_c4) # (8,2,8,8)
        outputs.append(p_c4)
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False) #(8,256,64,64)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3]) #(8,256,16,16)
        _c3   = self.ceff2([_c3_1, _c3_2]) # (8,256,16,16)
        p_c3  = self.make_pred_c3(_c3) # (8,2,16,16)
        outputs.append(p_c3)
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False) #(8,256,64,64)
        
        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.ceff3([_c2_1, _c2_2]) # (8,256,32,32)
        p_c2  = self.make_pred_c2(_c2) # (8,2,32,32)
        outputs.append(p_c2)
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False) #(8,256,64,64)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.ceff4([_c1_1, _c1_2])
        p_c1  = self.make_pred_c1(_c1)
        outputs.append(p_c1) 

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat([_c4_up, _c3_up, _c2_up, _c1],dim=1))
        # _c = self.linear_fuse(torch.cat([_c4_up, _c3_up],dim=1)) #(8,256,64,64)

        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        #Residual block
        x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #Residual block
        x = self.dense_1x(x) #(8,256,256,256)
        #Final prediction
        cp = self.change_probability(x) #(8,2,256,256)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs   # output not compared with gt_labels yet!

class MLPDecoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, align_corners=True, in_channels=[64, 128, 320, 512], embedding_dim=256, 
                 patch_size = 2, decoder_type = 'decoderA', output_nc=2, decoder_softmax=False):
        super(MLPDecoder, self).__init__()
        
        #settings
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim  # output.dim = input.dim
        self.output_nc       = output_nc
        self.patch_size = patch_size
        self.decoder_type = decoder_type
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP Mixer
        img_res = [8,16,32,64]
        self.mlp_mix4 = MLPMixer(image_size=img_res[2], channels=c2_in_channels,
                                 patch_size=patch_size, dim=512, depth=1)
        self.mlp_mix3 = MLPMixer(image_size=img_res[2], channels=c2_in_channels,
                                 patch_size=patch_size, dim=512, depth=1)
        self.share_mixer = MLPMixer(image_size=img_res[2],channels=c2_in_channels,
                                    patch_size=patch_size,dim=512,depth=1)
        
        self.linear4 = MLP(input_dim=c4_in_channels, embed_dim=c2_in_channels)
        self.linear3 = MLP(input_dim=c3_in_channels, embed_dim=c2_in_channels)

        desired_in_channels = int(embedding_dim*len(in_channels)/2)
        self.linear_fuse = nn.Sequential(
           nn.Conv2d(in_channels=desired_in_channels, out_channels=self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid()

    def forward(self, x_1, x_2):

        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1  # C1 (8,64,64,64), C2 (8,128,32,32)
        c1_2, c2_2, c3_2, c4_2 = x_2  # C3 (8,320,16,16), C4 (8,512,8,8)

        ############## MLP decoder on C3-C4 ###########
        n4, c4, h4, w4 = c4_1.shape
        n3, c3, h3, w3 = c3_1.shape

        outputs = []

        #### Difference Module
        ## Subtraction
        feats4_sub = torch.abs(c4_1 - c4_2)
        feats3_sub = torch.abs(c3_1 - c3_2)

        ##### Reshape (compress channel, expand resolution)
        feats4 = self.linear4(feats4_sub).permute(0,2,1).reshape(n4,-1,w4,h4) #(8,128,8,8)
        feats3 = self.linear3(feats3_sub).permute(0,2,1).reshape(n3,-1,w3,h3) #(8,128,16,16)

        feats4 = resize(feats4, size=c2_1.size()[2:], mode='bilinear', align_corners=False) #(8,128,32,32)
        feats3 = resize(feats3, size=c2_1.size()[2:], mode='bilinear', align_corners=False) #(8,128,32,32)
    
        if self.decoder_type == 'decoderA':
            feats4_mlp = self.share_mixer(feats4)
            feats3_mlp = self.share_mixer(feats3)
            feats = self.linear_fuse(torch.cat([feats4_mlp, feats3_mlp], dim=1))

        elif self.decoder_type == 'decoderB':
            feats4_mlp = self.mlp_mix4(feats4)
            feats3_mlp = self.mlp_mix3(feats3)
            feats = self.linear_fuse(torch.cat([feats4_mlp, feats3_mlp], dim=1))

        elif self.decoder_type == 'decoderC':
            fused_feats = self.linear_fuse(torch.cat((feats4, feats3), dim=1))
            feats = self.share_mixer(fused_feats)
        
        else:
            raise NotImplementedError('Decoder type [%s] is not supported' % self.decoder_type)

        x = feats

        #Upsampling #(8,128,32,32) -> (8,128,128,128)
        for i in range(2):
            x = self.convd2x(x)
            x = self.dense_2x(x)

        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #Residual block
        x = self.dense_1x(x) #(8,128,256,256)

        #Logits
        logits = self.change_probability(x) #(8,2,256,256)

        outputs.append(logits)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs   # output not compared with gt_labels yet!



# ScratchFormer:
class ScratchFormer(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256, patch_size=2, decoder_type='decoderA'):
        super(ScratchFormer, self).__init__()
        #Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths     = [3, 3, 9, 3]
        self.embedding_dim = embed_dim
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 
        self.patch_size = patch_size
        self.decoder_type = decoder_type

        self.Tenc_x2    = EncoderTransformer(patch_size = 7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                                             attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate,
                                             norm_layer=partial(LayerNorm, eps=1e-6), depths=self.depths)
        
        #Transformer Decoder
        self.TDec_x1   = DecoderTransformer(align_corners=False, in_channels = self.embed_dims, embedding_dim= self.embedding_dim,
                                            output_nc=output_nc, decoder_softmax = decoder_softmax)
        
        self.TDec_x2   = MLPDecoder(align_corners=False, in_channels = self.embed_dims, embedding_dim= self.embedding_dim,
                                    patch_size= self.patch_size, decoder_type = self.decoder_type,
                                    output_nc=output_nc, decoder_softmax = decoder_softmax)

    def forward(self, x1, x2):

        # x1 (8,3,256,256)
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]

        if self.decoder_type == 'base':
            cp = self.TDec_x1(fx1, fx2)
        else: 
            cp = self.TDec_x2(fx1, fx2)
        return cp
