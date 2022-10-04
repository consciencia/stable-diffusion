import os
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
if os.name != "nt":
    import xformers
    import xformers.ops
from typing import Any, Optional

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        o = self.net(x)
        del x
        return o


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        del x
        q, k1, v = rearrange(qkv,
                            'b (qkv heads c) h w -> qkv b heads c (h w)',
                            heads=self.heads,
                            qkv=3)
        del qkv
        k2 = k1.softmax(dim=-1)
        del k1
        context = torch.einsum('bhdn,bhen->bhde', k2, v)
        del k2, v
        out1 = torch.einsum('bhde,bhdn->bhen', context, q)
        del context, q
        out2 = rearrange(out1,
                         'b heads c (h w) -> b (heads c) h w',
                         heads=self.heads,
                         h=h,
                         w=w)
        del out1
        out3 = self.to_out(out2)
        del out2
        return out3


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., att_step=1):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.att_step = att_step

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                    nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        del context, x

        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                         (q, k, v))
        del q, k, v


        limit = k1.shape[0]
        att_step = self.att_step
        q_chunks = list(torch.tensor_split(q1, limit // att_step, dim=0))
        k_chunks = list(torch.tensor_split(k1, limit // att_step, dim=0))
        v_chunks = list(torch.tensor_split(v1, limit // att_step, dim=0))
        q_chunks.reverse()
        k_chunks.reverse()
        v_chunks.reverse()

        sim = torch.zeros(q1.shape[0],
                          q1.shape[1],
                          v1.shape[2],
                          device=q1.device)
        del k1, q1, v1

        for i in range (0, limit, att_step):
            q_buffer = q_chunks.pop()
            k_buffer = k_chunks.pop()
            v_buffer = v_chunks.pop()
            sim_buffer = einsum('b i d, b j d -> b i j',
                                q_buffer,
                                k_buffer) * self.scale
            del k_buffer, q_buffer

            # attention, what we cannot get enough of, by chunks
            sim_buffer1 = sim_buffer.softmax(dim=-1)
            del sim_buffer

            sim_buffer2 = einsum('b i j, b j d -> b i d',
                                 sim_buffer1,
                                 v_buffer)
            del sim_buffer1, v_buffer

            sim[i:i+att_step,:,:] = sim_buffer2
            del sim_buffer2

        sim1 = rearrange(sim, '(b h) n d -> b n (h d)', h=h)
        del sim
        sim2 = self.to_out(sim1)
        del sim1
        return sim2


class MemoryEfficientCrossAttention(nn.Module):
     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
         super().__init__()
         inner_dim = dim_head * heads
         context_dim = default(context_dim, query_dim)

         self.heads = heads
         self.dim_head = dim_head

         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

         self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
         self.attention_op: Optional[Any] = None

     def forward(self, x, context=None, mask=None):
         q_in = self.to_q(x)
         context = default(context, x)
         k_in = self.to_k(context)
         v_in = self.to_v(context)
         del context, x

         b, _, _ = q_in.shape
         q, k, v = map(
             lambda t: t.unsqueeze(3)
             .reshape(b, t.shape[1], self.heads, self.dim_head)
             .permute(0, 2, 1, 3)
             .reshape(b * self.heads, t.shape[1], self.dim_head)
             .contiguous(),
             (q_in, k_in, v_in),
         )
         del q_in, k_in, v_in

         # actually compute the attention, what we cannot get enough of
         out = xformers.ops.memory_efficient_attention(q,
                                                       k,
                                                       v,
                                                       attn_bias=None,
                                                       op=self.attention_op)
         del q, k, v

         # TODO: Use this directly in the attention operation, as a bias
         if exists(mask):
             raise NotImplementedError()
         out2 = (out.unsqueeze(0)
                 .reshape(b, self.heads, out.shape[1], self.dim_head)
                 .permute(0, 2, 1, 3)
                 .reshape(b, out.shape[1], self.heads * self.dim_head))
         del out

         out3 = self.to_out(out2)
         del out2

         return out3


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        if os.name == "nt":
            ctor = CrossAttention
        else:
            ctor = MemoryEfficientCrossAttention
        # is a self-attention
        self.attn1 = ctor(query_dim=dim,
                          heads=n_heads,
                          dim_head=d_head,
                          dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # is self-attn if context is none
        self.attn2 = ctor(query_dim=dim,
                          context_dim=context_dim,
                          heads=n_heads,
                          dim_head=d_head,
                          dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def forward(self, x, context=None):
        return checkpoint(self._forward,
                          (x, context),
                          self.parameters(),
                          self.checkpoint)

    def _forward(self, x, context=None):
        x = x.contiguous() if x.device.type == "mps" else x
        x1 = self.attn1(self.norm1(x))
        x1 += x
        del x
        x2 = self.attn2(self.norm2(x1), context=context)
        x2 += x1
        del x1
        x3 = self.ff(self.norm3(x2))
        x3 += x2
        del x2
        return x3


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim,
                                   n_heads,
                                   d_head,
                                   dropout=dropout,
                                   context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x1 = self.norm(x)
        x2 = self.proj_in(x1)
        del x1
        x3 = rearrange(x2, 'b c h w -> b (h w) c')
        del x2
        for block in self.transformer_blocks:
            t = x3
            x3 = block(t, context=context)
            del t
        x4 = rearrange(x3, 'b (h w) c -> b c h w', h=h, w=w)
        del x3
        x5 = self.proj_out(x4)
        del x4
        x5 += x_in
        del x_in
        return x5
