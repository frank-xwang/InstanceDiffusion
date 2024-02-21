from abc import abstractmethod

import numpy as np
import random
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from torch.utils import checkpoint
from ldm.util import instantiate_from_config
from copy import deepcopy


def Fourier_filter(x_in, threshold, scale):
    x = x_in
    B, C, H, W = x.shape

    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.to(dtype=torch.float32)

    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype=x_in.dtype)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context, objs, grounding_input=None, drop_box_mask=False):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                if grounding_input is not None:
                    x = layer(x, context, objs, grounding_input, drop_box_mask=drop_box_mask)
                else:
                    x = layer(x, context, objs, drop_box_mask=drop_box_mask)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x




class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, emb )
        else:
            return self._forward(x, emb) 


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


def preprocess_text(processor, input):
    if input == None:
        return None
    inputs = processor(text=input,  return_tensors="pt", padding=True)
    inputs['input_ids'] = inputs['input_ids'].cuda()
    inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
    inputs['attention_mask'] = inputs['attention_mask'].cuda()
    return inputs

def get_clip_feature_text(model, processor, input):
    which_layer_text = 'before'
    inputs = preprocess_text(processor, input)
    if inputs == None:
        return None
    outputs = model(**inputs)
    if which_layer_text == 'before':
        feature = outputs.text_model_output.pooler_output
    return feature

class ScaleULayer(nn.Module):
    def __init__(self, channel, reduction=16, out_channels=None):
        super(ScaleULayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if out_channels is None:
            self.out_channels = channel
        else:
            self.out_channels = out_channels
        self.scaleu_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, self.out_channels, bias=False),
        )
        self.scaleu_fc.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.0)
            # nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.scaleu_fc(y).view(b, self.out_channels)
        if self.out_channels == 1:
            return y.view(b, 1, 1, 1)
        return y

class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=8,
        use_scale_shift_norm=False,
        transformer_depth=1,           
        context_dim=None,  
        fuser_type = None,
        inpaint_mode = False,
        grounding_downsampler = None,
        grounding_tokenizer = None,
        sd_v1_5 = False,
        efficient_attention = False,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.fuser_type = fuser_type
        self.inpaint_mode = inpaint_mode
        self.sd_v1_5 = sd_v1_5
        assert fuser_type in ["gatedSA","gatedSA2","gatedCA"]
        self.efficient_attention = efficient_attention

        self.grounding_tokenizer_input = None # set externally

        self.enable_freeu = False # set externally
        self.enable_scaleu = True # True
        assert self.enable_freeu != self.enable_scaleu
        self.enable_se_scaleu = False

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.downsample_net = None 
        self.additional_channel_from_downsampler = 0
        self.first_conv_restorable = True 

        in_c = in_channels+self.additional_channel_from_downsampler
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_c, model_channels, 3, padding=1))])


        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        # = = = = = = = = = = = = = = = = = = = = Down Branch = = = = = = = = = = = = = = = = = = = = #
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ ResBlock(ch,
                                    time_embed_dim,
                                    dropout,
                                    out_channels=mult * model_channels,
                                    dims=dims,
                                    use_checkpoint=use_checkpoint,
                                    use_scale_shift_norm=use_scale_shift_norm,) ]

                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(SpatialTransformer(ch, key_dim=context_dim, value_dim=context_dim, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, fuser_type=fuser_type, use_checkpoint=use_checkpoint, efficient_attention=efficient_attention))
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1: # will not go to this downsample branch in the last feature
                out_ch = ch
                self.input_blocks.append( TimestepEmbedSequential( Downsample(ch, conv_resample, dims=dims, out_channels=out_ch ) ) )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
        dim_head = ch // num_heads

        # self.input_blocks = [ C |  RT  RT  D  |  RT  RT  D  |  RT  RT  D  |   R  R   ]

        # = = = = = = = = = = = = = = = = = = = = BottleNeck = = = = = = = = = = = = = = = = = = = = #
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            SpatialTransformer(ch, key_dim=context_dim, value_dim=context_dim, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, fuser_type=fuser_type, use_checkpoint=use_checkpoint, efficient_attention=efficient_attention),
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm))



        # = = = = = = = = = = = = = = = = = = = = Up Branch = = = = = = = = = = = = = = = = = = = = #


        self.output_blocks = nn.ModuleList([])
        
        idx = 0
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ ResBlock(ch + ich,
                                    time_embed_dim,
                                    dropout,
                                    out_channels=model_channels * mult,
                                    dims=dims,
                                    use_checkpoint=use_checkpoint,
                                    use_scale_shift_norm=use_scale_shift_norm) ]
                if self.enable_scaleu:
                    self.register_parameter( 'scaleu_b_{}'.format(idx), nn.Parameter(torch.zeros(ch)) )
                    self.register_parameter( 'scaleu_s_{}'.format(idx), nn.Parameter(torch.zeros(1)) )
                idx += 1

                ch = model_channels * mult

                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append( SpatialTransformer(ch, key_dim=context_dim, value_dim=context_dim, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, fuser_type=fuser_type, use_checkpoint=use_checkpoint, efficient_attention=efficient_attention) )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append( Upsample(ch, conv_resample, dims=dims, out_channels=out_ch) )
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # self.output_blocks = [ R  R  RU | RT  RT  RTU |  RT  RT  RTU  |  RT  RT  RT  ]

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        
        # grounding_tokenizer: ldm.modules.diffusionmodules.text_grounding_net.UniFusion
        self.position_net = instantiate_from_config(grounding_tokenizer)

    def restore_first_conv_from_SD(self):
        if self.first_conv_restorable:
            device = self.input_blocks[0][0].weight.device

            if not self.sd_v1_5:
                SD_weights = th.load("pretrained/SD_input_conv_weight_bias.pth")
            else:
                SD_weights = th.load("pretrained/SD_v1_5_input_conv_weight_bias.pth")
            self.first_conv_state_dict = deepcopy(self.input_blocks[0][0].state_dict())
            self.input_blocks[0][0] = conv_nd(2, 4, 320, 3, padding=1)
            self.input_blocks[0][0].load_state_dict(SD_weights)
            self.input_blocks[0][0].to(device)

    def forward_single_input(self, input):
        if ("grounding_input" in input):
            grounding_input = input["grounding_input"]
        else: 
            # Guidance null case
            grounding_input = self.grounding_tokenizer_input.get_null_input()

        if self.training and random.random() < 0.1 and self.grounding_tokenizer_input.set: # random drop for guidance  
            batch_size = grounding_input['boxes'].size(0)
            grounding_input = self.grounding_tokenizer_input.get_null_input(batch=batch_size)

        # Grounding tokens: B*N*C
        objs, drop_box_mask = self.position_net( grounding_input['boxes'], grounding_input['masks'], grounding_input['positive_embeddings'], grounding_input['scribbles'], grounding_input['polygons'], grounding_input['segs'], grounding_input['points'] )

        # Time embedding 
        t_emb = timestep_embedding(input["timesteps"], self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        # input tensor  
        h = input["x"]

        # Text input 
        context = input["context"]

        # Start forwarding 
        hs = []
        for module in self.input_blocks:
            if 'att_masks' in grounding_input:
                h = module(h, emb, context, objs, grounding_input, drop_box_mask=drop_box_mask)
            else:
                h = module(h, emb, context, objs, drop_box_mask=drop_box_mask)
            hs.append(h)

        if 'att_masks' in grounding_input:
            h = self.middle_block(h, emb, context, objs, grounding_input, drop_box_mask=drop_box_mask)
        else:
            h = self.middle_block(h, emb, context, objs, drop_box_mask=drop_box_mask)

        for idx, module in enumerate(self.output_blocks):
            if self.enable_scaleu:
                # --------------- ScaleU code -----------------------
                # operate on all stages and all channels
                hs_ = hs.pop()
                b = torch.tanh(getattr(self, 'scaleu_b_{}'.format(idx)) ) + 1
                s = torch.tanh(getattr(self, 'scaleu_s_{}'.format(idx)) ) + 1
                if self.enable_se_scaleu:
                    hidden_mean = h.mean(1).unsqueeze(1) # B,1,H,W 
                    B = hidden_mean.shape[0]
                    hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) # B,1
                    hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True) # B,1
                    # duplicate the hidden_mean dimension 1 to C
                    hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3) # B,1,H,W
                    b = torch.einsum('c,bchw->bchw', b-1, hidden_mean) + 1.0 # B,C,H,W
                    h = torch.einsum('bchw,bchw->bchw', h, b)
                else:      
                    h = torch.einsum('bchw,c->bchw', h, b)
                hs_ = Fourier_filter(hs_, threshold=1, scale=s)
                # ---------------------------------------------------------
                h = th.cat([h, hs_], dim=1)
            elif self.enable_freeu:
                b1 = 1.2
                b2 = 1.4
                s1 = 0.9
                s2 = 0.2
                hs_ = hs.pop()
                # --------------- FreeU code -----------------------
                # Only operate on the first two stages
                if h.shape[1] == 1280:
                    h[:,:640] = h[:,:640] * b1
                    hs_ = Fourier_filter(hs_, threshold=1, scale=s1)
                if h.shape[1] == 640:
                    h[:,:320] = h[:,:320] * b2
                    hs_ = Fourier_filter(hs_, threshold=1, scale=s2)
                # ---------------------------------------------------------
                h = th.cat([h, hs_], dim=1)
            else:
                h = th.cat([h, hs.pop()], dim=1)
            if 'att_masks' in grounding_input:
                h = module(h, emb, context, objs, grounding_input, drop_box_mask=drop_box_mask)
            else:
                h = module(h, emb, context, objs, drop_box_mask=drop_box_mask)

        return self.out(h)

    def forward(self, input):
        return self.forward_single_input(input)
