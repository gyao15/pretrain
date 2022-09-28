import copy
import warnings
import numpy as np
import torch
import mmcv
import os
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr, limit_period, PseudoSampler)
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_batch


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, attn_mask=None):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = key_pos

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed), attn_mask=attn_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,  # type: Optional[Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[Tensor]
                                 k_proj_weight=None,  # type: Optional[Tensor]
                                 v_proj_weight=None,  # type: Optional[Tensor]
                                 static_k=None,  # type: Optional[Tensor]
                                 static_v=None,  # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 **kwargs):
        super(FFN, self).__init__()

        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict

class ContrastiveFeatureHead(nn.Module):
    def __init__(self, num_conv, num_classes, dropout=0.1, t=0.2):
        super(ContrastiveFeatureHead, self).__init__()
        self.num_classes = num_classes

        self.k_layers = nn.ModuleList()
        self.q_layers = nn.ModuleList()
        self.v_layers = nn.ModuleList()
        self.t = t
        for i in range(self.num_classes):
            self.k_layers.append(nn.Conv1d(num_conv, num_conv, 1, 1))
            self.q_layers.append(nn.Conv1d(num_conv, num_conv, 1, 1))
            self.v_layers.append(nn.Conv1d(num_conv, num_conv, 1, 1))

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.out_layer = nn.Sequential(nn.Conv2d(num_conv, num_conv, 1, 1), nn.ReLU())
        self.bn1 = nn.BatchNorm2d(num_conv)
        self.bn2 = nn.BatchNorm2d(num_conv)

    def forward(self, query, key):
        outs = []
        sims = []
        for i in range(self.num_classes):
            out = query[..., i]
            for j in range(1):
                q = self.q_layers[i](out)
                k = self.k_layers[i](key)
                v = self.v_layers[i](key)
                sim = torch.matmul(q.transpose(1, 2), k) / self.t
                sim = torch.softmax(sim, dim=1)
                out = torch.sum(sim.unsqueeze(1) * v.unsqueeze(2), dim=-1)
                #out = torch.sum(sim.unsqueeze(1) * v.unsqueeze(2), dim=-1) / torch.clamp(torch.sum(sim.unsqueeze(1), dim=-1), min=1e-6)
            outs.append(out)
            sims.append(sim)
        outs = torch.stack(outs, dim=-1)
        sims = torch.stack(sims, dim=-1)
        outs = query + self.drop1(outs)
        outs = self.out_layer(self.bn1(outs))
        outs = query + self.drop2(outs)
        outs = self.bn2(outs)
        return outs, sims



class SFFN(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 **kwargs):
        super(SFFN, self).__init__()

        self.heads = heads
        self.init_bias = init_bias
        self.trainable = kwargs.get('trainable', True)
        for head in self.heads:
            classes, num_conv, num_bins, num_encoder, num_decoder, num_heads, ffn_channel, activation = self.heads[head]
            conv_layers = nn.ModuleList()
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv
            self.__setattr__(head+'_conv_layers', nn.Sequential(*conv_layers))

            if head == 'heatmap':
                self.__setattr__(head+'_embed', nn.Parameter(torch.randn(head_conv, num_bins+1, classes), requires_grad=self.trainable))
            else:
                self.__setattr__(head+'_embed', nn.Parameter(torch.randn(head_conv, num_bins, classes), requires_grad=self.trainable))
            decoder = nn.ModuleList()
            for i in range(num_decoder):
                decoder.append(
                    ContrastiveFeatureHead(head_conv, classes))
            self.__setattr__(head+'_decoder', decoder)
            #print(self.__getattr__(head+'_decoder'))

            encoder = nn.ModuleList()
            for i in range(num_encoder):
                if head == 'heatmap':
                    encoder.append(
                        ContrastiveTransformerLayer(
                            head_conv, num_heads, ffn_channel, num_bins, dropout=0.1, activation=activation,
                            self_posembed=PositionEmbeddingLearned(2, head_conv),
                        )
                    )
                else:
                    encoder.append(
                        ContrastiveTransformerLayer(
                            head_conv, num_heads, ffn_channel, classes, dropout=0.1, activation=activation,
                            self_posembed=PositionEmbeddingLearned(2, head_conv),
                        )
                    )
            self.__setattr__(head+'_encoder', encoder)
            if head == 'heatmap':
                out_layer = build_conv_layer(
                        dict(type='Conv1d'),
                        head_conv,
                        num_bins,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True)
                self.__setattr__(head, out_layer)
                embed_layer = build_conv_layer(
                        dict(type='Conv1d'),
                        num_bins,
                        head_conv,
                        kernel_size=1,
                        stride=1,
                        bias=True)
                self.__setattr__(head+'_embed_layer', embed_layer)
            else:
                out_layer = build_conv_layer(
                        dict(type='Conv1d'),
                        head_conv,
                        classes,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True)
                self.__setattr__(head, out_layer)
                out_var_layer = build_conv_layer(
                        dict(type='Conv1d'),
                        head_conv,
                        classes,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True)
                self.__setattr__(head+'_var', out_var_layer)
                embed_layer = build_conv_layer(
                        dict(type='Conv1d'),
                        classes,
                        head_conv,
                        kernel_size=1,
                        stride=1,
                        bias=True)
                self.__setattr__(head+'_embed_layer', embed_layer)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head).bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head+'_conv_layers').modules():
                    if isinstance(m, nn.Conv1d):
                        kaiming_init(m)
                kaiming_init(self.__getattr__(head))
                kaiming_init(self.__getattr__(head+'_var'))
                self.__getattr__(head+'_var').bias.data.fill_(1)

    def with_embed(self, x, e):
        return x if e is None else x + e

    def forward(self, x, x_pos, preds_dict=None, gt_head_dict=None):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        batch_size = x.shape[0]
        for head in self.heads:
            head_related_features = self.__getattr__(head+'_conv_layers')(x)
            statistic_embed = self.__getattr__(head+'_embed')
            if preds_dict is not None: 
                head_embed = self.__getattr__(head+'_embed_layer')(preds_dict[head])
            else:
                head_embed = None
            num_bins, num_classes = statistic_embed.shape[-2:]
            statistic_embed = statistic_embed.unsqueeze(0).repeat(batch_size, 1, 1, 1)

            #print(head, torch.isnan(statistic_embed).sum())
            for i in range(len(self.__getattr__(head+'_decoder'))):
                statistic_embed, sims = self.__getattr__(head+'_decoder')[i](statistic_embed, self.with_embed(head_related_features, head_embed))
            ret_dict[head+'_sims'] = sims
            if not self.trainable:
                t = 0.95 * self.__getattr__(head+'_embed') + 0.05 * statistic_embed.detach().clone()
                self.__setattr__(head+'_embed',  t)
            statistic_embed = statistic_embed.view(batch_size, -1, num_bins*num_classes)
            #print(head)
            #print(torch.isnan(statistic_embed).sum())
            for i in range(len(self.__getattr__(head+'_encoder'))):
                #is_last = (i==(len(self.__getattr__(head+'_encoder'))-1))
                is_last = False
                head_related_features, head_related_features_p, attn_map_x, attn_map_xy = self.__getattr__(head+'_encoder')[i](statistic_embed, 
                                                                            self.with_embed(head_related_features, head_embed), x_pos, gt_head_dict[head], num_classes, is_last)
            
            
            #print(torch.isnan(head_related_features).sum())
            ret_dict[head+'_mean'] = self.__getattr__(head)(head_related_features)
            if head_related_features_p is not None:
                ret_dict[head+'_p'] = self.__getattr__(head)(head_related_features_p)

            if is_last:
                attn_map_x = attn_map_x.mean(-1)
                if attn_map_xy is not None:
                    attn_map_xy = attn_map_xy.mean(-1)
            else:
                attn_map_x = attn_map_x.view(batch_size, x.shape[-1], num_bins, -1)
                if attn_map_xy is not None:
                    attn_map_xy = attn_map_xy.view(batch_size, x.shape[-1], num_bins, -1)

            ret_dict[head+'_attn_map_x'] = attn_map_x
            ret_dict[head+'_attn_map_xy'] = attn_map_xy
        
            if head != 'heatmap':
                ret_dict[head+'_var'] = self.__getattr__(head+'_var')(head_related_features)
                '''
                attn_map_x = attn_map_x.permute(0, 3, 1, 2)
                if attn_map_xy is not None:
                    attn_map_xy = attn_map_xy.permute(0, 3, 1, 2)
                    ret_dict[head] = torch.sum(attn_map_xy * ret_dict[head+'_mean'], dim=-1)
                else:
                    ret_dict[head] = torch.sum(attn_map_x * ret_dict[head+'_mean'], dim=-1)
                '''
                '''
                if attn_map_xy is not None:
                    attn_map_xy = attn_map_xy.squeeze(-1).unsqueeze(1)
                    ret_dict[head] = torch.sum(attn_map_xy * ret_dict[head+'_mean'], dim=-1)
                else:
                    ret_dict[head] = torch.sum(attn_map_x * ret_dict[head+'_mean'], dim=-1)
                '''
            
        return ret_dict

class ContrastiveAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, gt_dim, bias=True):
        super(ContrastiveAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = float(self.head_dim) ** -0.5
        self.bias = bias

        self.gt_proj = nn.Linear(gt_dim, embed_dim, bias=bias)
        self.in_proj_q_x = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_k_x = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_q_xy = nn.Linear(embed_dim*2, embed_dim, bias=bias)
        self.in_proj_k_xy = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if self.bias:
                    constant_(m.bias, 0.)

    def forward(self, head_embed, features, gt_infos, num_classes, is_last):
        batch_size = head_embed.shape[0]
        N = features.shape[1]
        M = head_embed.shape[1]
        q_x = self.in_proj_q_x(features)
        k_x = self.in_proj_k_x(head_embed)
        q_x = q_x.view(batch_size, -1, self.num_heads, self.head_dim)
        k_x = k_x.view(batch_size, -1, self.num_heads, self.head_dim)
        attn_map_x = torch.einsum('bnhc,bmhc->bnmh', q_x, k_x)
        attn_map_x = attn_map_x.view(batch_size, N, -1, num_classes, self.num_heads)
        attn_map_x = torch.softmax(attn_map_x * self.scale, dim=2)

        v = self.in_proj_v(head_embed)
        v = v.view(batch_size, 1, -1, num_classes, self.num_heads, self.head_dim)

        if self.training and gt_infos is not None:
            #print(gt_infos.shape)
            gt_emb = self.gt_proj(gt_infos)
            q_xy = self.in_proj_q_xy(torch.cat([features, gt_emb], dim=-1))
            k_xy = self.in_proj_k_xy(head_embed)
            q_xy = q_xy.view(batch_size, -1, self.num_heads, self.head_dim)
            k_xy = k_xy.view(batch_size, -1, self.num_heads, self.head_dim)
            attn_map_xy = torch.einsum('bnhc,bmhc->bnmh', q_xy, k_xy)
            attn_map_xy = attn_map_xy.view(batch_size, N, -1, num_classes, self.num_heads)
            attn_map_xy = torch.softmax(attn_map_xy * self.scale, dim=2)
            out = attn_map_xy.unsqueeze(-1) * v
            out = torch.sum(out, dim=3)
            if not is_last:
                out = torch.sum(out, dim=2).contiguous().view(batch_size, N, -1)
            else:
                out = out.contiguous().view(batch_size, N, -1, self.num_heads*self.head_dim)
            out = self.out_proj(out)

            out_p = attn_map_x.unsqueeze(-1) * v
            out_p = torch.sum(out_p, dim=3)
            if not is_last:
                out_p = torch.sum(out_p, dim=2).contiguous().view(batch_size, N, -1)
            else:
                out_p = out_p.contiguous().view(batch_size, N, -1, self.num_heads*self.head_dim)
            out_p = self.out_proj(out_p)
            #print(torch.isnan(q_x).sum(), torch.isnan(q_xy).sum(), torch.isnan(k_x).sum(), torch.isnan(k_xy).sum(), torch.isnan(v).sum())
            return out, out_p, attn_map_x, attn_map_xy
        else:
            attn_map_xy = None
            out = attn_map_x.unsqueeze(-1) * v
            out = torch.sum(out, dim=3)
            if not is_last:
                out = torch.sum(out, dim=2).contiguous().view(batch_size, N, -1)
            else:
                out = out.contiguous().view(batch_size, N, -1, self.num_heads*self.head_dim)
            out = self.out_proj(out)
            return out, None, attn_map_x, attn_map_xy

class ContrastiveTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_channel, gt_dim, dropout=0.0, activation='relu', self_posembed=None, bias=True):
        super().__init__()
        self.contrastive_attn = ContrastiveAttention(embed_dim, num_heads, gt_dim, bias=bias)
        self.pos_embed = self_posembed
        self.linear1 = nn.Linear(embed_dim, ffn_channel)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_channel, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed
    
    def forward(self, head_embed, features, pos, gt_infos, num_classes, is_last):
        """
        :param query: B C Pq
        :param key: B Pk C
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
        """

        features = features.transpose(1, 2)
        #pos_emb = self.pos_embed(pos).transpose(1, 2)
        pos_emb = None
        head_embed = head_embed.transpose(1, 2)
        if gt_infos is not None:
            gt_infos = gt_infos.transpose(1, 2)
        
        feat, feat_p, attn_map_x, attn_map_xy = self.contrastive_attn(head_embed, self.with_pos_embed(features, pos_emb), gt_infos, num_classes, is_last)
        if is_last:
            x = features.unsqueeze(2).repeat(1, 1, feat.shape[2], 1) + self.dropout2(feat)
        else:
            #print(features.shape, feat.shape)
            x = features + self.dropout2(feat)
        x = self.norm2(x)

        features2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(features2)
        x = self.norm3(x)

        
        # BxPqxPkxC
        if is_last:
            x = x.permute(0, 3, 1, 2)
        else:
            x = x.transpose(1, 2)

        if feat_p is not None:
            if is_last:
                x_p = features.unsqueeze(2).repeat(1, 1, feat.shape[2], 1) + self.dropout2(feat_p)
            else:
                x_p = features + self.dropout2(feat_p)
            x_p = self.norm2(x_p)

            features2 = self.linear2(self.dropout(self.activation(self.linear1(x_p))))
            x_p = x_p + self.dropout3(features2)
            x_p = self.norm3(x_p)

            # BxPqxPkxC
            if is_last:
                x_p = x_p.permute(0, 3, 1, 2)
            else:
                x_p = x_p.transpose(1, 2)
        else:
            x_p = None
        return x, x_p, attn_map_x, attn_map_xy

@HEADS.register_module()
class STransFusionHeadv2(nn.Module):
    def __init__(self,
                 fuse_img=False,
                 num_views=0,
                 in_channels_img=64,
                 out_size_factor_img=4,
                 num_proposals=128,
                 auxiliary=True,
                 in_channels=128 * 3,
                 hidden_channel=128,
                 num_classes=4,
                 # config for Transformer
                 num_decoder_layers=3,
                 num_heads=8,
                 learnable_query_pos=False,
                 initialize_by_heatmap=False,
                 nms_kernel_size=1,
                 ffn_channel=256,
                 dropout=0.1,
                 bn_momentum=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_iou=dict(type='VarifocalLoss', use_sigmoid=True, iou_weighted=True, reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_pbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_extra=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_dist=dict(type='KLLoss', loss_weights=0.25),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 statistic_cfg=None,
                 trainable=True,
                 ):
        super(STransFusionHeadv2, self).__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.learnable_query_pos = learnable_query_pos
        self.initialize_by_heatmap = initialize_by_heatmap
        self.nms_kernel_size = nms_kernel_size
        if self.initialize_by_heatmap is True:
            assert self.learnable_query_pos is False, "initialized by heatmap is conflicting with learnable query position"
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_dist = build_loss(loss_dist)
        self.loss_pbox = build_loss(loss_pbox)
        self.loss_bbox_type = loss_bbox['type']

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False
        self.hidden_channel = hidden_channel

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        if self.initialize_by_heatmap:
            if statistic_cfg['heatmap_head']:
                self.heatmap_conv_layers = ConvModule(
                    hidden_channel,
                    hidden_channel,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN2d'),
                )
                head = dict(heatmap=copy.deepcopy(common_heads['heatmap']))
                self.headmap_head = SFFN(hidden_channel, head, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias, trainable=trainable)
            else:
                layers = []
                layers.append(ConvModule(
                    hidden_channel,
                    hidden_channel,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN2d'),
                ))
                layers.append(build_conv_layer(
                    dict(type='Conv2d'),
                    hidden_channel,
                    num_classes,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                ))
                self.heatmap_head = nn.Sequential(*layers)
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        else:
            # query feature
            self.query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2]), requires_grad=learnable_query_pos)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                ))

        # Prediction Head
        self.statistic_cfg = statistic_cfg
        self.prediction_heads = nn.ModuleList()
        self.extra_prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            self.prediction_heads.append(SFFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias, trainable=trainable))
            if self.statistic_cfg['extra_head'] is not None:
                extra_heads = copy.deepcopy(self.statistic_cfg['extra_head'])
                self.extra_prediction_heads.append(FFN(hidden_channel, extra_heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))
                #self.loss_extra = build_loss(loss_extra)
            
        self.fuse_img = fuse_img
        if self.fuse_img:
            self.num_views = num_views
            self.out_size_factor_img = out_size_factor_img
            self.shared_conv_img = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_img,  # channel of img feature map
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
            self.heatmap_head_img = copy.deepcopy(self.heatmap_head)
            # transformer decoder layers for img fusion
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                ))
            # cross-attention only layers for projecting img feature onto BEV
            for i in range(num_views):
                self.decoder.append(
                    TransformerDecoderLayer(
                        hidden_channel, num_heads, ffn_channel, dropout, activation,
                        self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                        cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                        cross_only=True,
                    ))
            self.fc = nn.Sequential(*[nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)])

            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(FFN(hidden_channel * 2, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs, img_inputs, img_metas, gt_bboxes_3d=None, gt_labels_3d=None):
        """Forward function for CenterPoint.

        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)

        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)
        

        #################################
        # image to BEV
        #################################
        #print(lidar_feat.shape)
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        if self.fuse_img:
            img_feat = self.shared_conv_img(img_inputs)  # [BS * n_views, C, H, W]

            img_h, img_w, num_channel = img_inputs.shape[-2], img_inputs.shape[-1], img_feat.shape[1]
            raw_img_feat = img_feat.view(batch_size, self.num_views, num_channel, img_h, img_w).permute(0, 2, 3, 1, 4) # [BS, C, H, n_views, W]
            img_feat = raw_img_feat.reshape(batch_size, num_channel, img_h, img_w * self.num_views)  # [BS, C, H, n_views*W]
            img_feat_collapsed = img_feat.max(2).values
            img_feat_collapsed = self.fc(img_feat_collapsed).view(batch_size, num_channel, img_w * self.num_views)

            # positional encoding for image guided query initialization
            if self.img_feat_collapsed_pos is None:
                img_feat_collapsed_pos = self.img_feat_collapsed_pos = self.create_2D_grid(1, img_feat_collapsed.shape[-1]).to(img_feat.device)
            else:
                img_feat_collapsed_pos = self.img_feat_collapsed_pos

            bev_feat = lidar_feat_flatten
            for idx_view in range(self.num_views):
                bev_feat = self.decoder[2 + idx_view](bev_feat, img_feat_collapsed[..., img_w * idx_view:img_w * (idx_view + 1)], bev_pos, img_feat_collapsed_pos[:, img_w * idx_view:img_w * (idx_view + 1)])

        #################################
        # image guided query initialization
        #################################
        if self.initialize_by_heatmap:
            if self.statistic_cfg['heatmap_head']:
                dense_heatmap_flatten = self.heatmap_conv_layers(lidar_feat).view(batch_size, lidar_feat.shape[1], -1)
                gt_dict = {'heatmap': None}
                dense_heatmap = self.headmap_head(dense_heatmap_flatten, bev_pos, None, gt_dict)['heatmap_mean']
                #print(dense_heatmap)
                dense_heatmap = dense_heatmap.view(batch_size, self.num_classes, lidar_feat.shape[-2], lidar_feat.shape[-1])
            else:
                dense_heatmap = self.heatmap_head(lidar_feat)
            dense_heatmap_img = None
            if self.fuse_img:
                dense_heatmap_img = self.heatmap_head_img(bev_feat.view(lidar_feat.shape))  # [BS, num_classes, H, W]
                heatmap = (dense_heatmap.detach().sigmoid() + dense_heatmap_img.detach().sigmoid()) / 2
            else:
                heatmap = dense_heatmap.detach().sigmoid()

            padding = self.nms_kernel_size // 2
            local_max = torch.zeros_like(heatmap)
            # equals to nms radius = voxel_size * out_size_factor * kenel_size
            local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
            ## for Pedestrian & Traffic_cone in nuScenes
            if self.test_cfg['dataset'] == 'nuScenes':
                local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            elif self.test_cfg['dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
                local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            elif self.test_cfg['dataset'] == 'once':
                local_max[:, 3, ] = F.max_pool2d(heatmap[:, 3], kernel_size=1, stride=1, padding=0)
                local_max[:, 4, ] = F.max_pool2d(heatmap[:, 4], kernel_size=1, stride=1, padding=0)
            heatmap = heatmap * (heatmap == local_max)
            heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

            # top #num_proposals among all classes
            top_proposals = heatmap.reshape(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
            top_proposals_class = top_proposals // heatmap.shape[-1]
            top_proposals_index = top_proposals % heatmap.shape[-1]
            query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
            self.query_labels = top_proposals_class

            # add category embedding
            one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
            query_cat_encoding = self.class_encoding(one_hot.float())
            query_feat += query_cat_encoding

            query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
            '''
            if self.training:
                with torch.no_grad():
                    query_gt_dict = self.get_gt_dict(gt_bboxes_3d, gt_labels_3d, bev_pos, top_proposals_index)
            else:
                heads = ['cls', 'center', 'height', 'dim', 'rot']
                query_gt_dict = {k: None for k in heads}
            '''
        else:
            query_feat = self.query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
            base_xyz = self.query_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, num_proposals, 2]

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)

            # Prediction
            res_layer = self.extra_prediction_heads[i](query_feat)
            
            first_res_layer = {}
            for k, v in res_layer.items():
                first_res_layer[k] = v.detach().clone()
                if k == 'heatmap':
                    first_res_layer[k] = torch.sigmoid(first_res_layer[k])

            if self.training:
                with torch.no_grad():
                    #print(res_layer['heatmap'])
                    labels, label_weights, bbox_targets, bbox_weights, _, _, _, _ = self.get_targets(gt_bboxes_3d, gt_labels_3d, [res_layer])
                    
                    bbox_targets[:, :, :2] -= query_pos
                    bbox_targets = bbox_targets.transpose(1, 2)
                    query_gt_dict = {
                        'center': bbox_targets[:, :2].detach().clone(),
                        'height': bbox_targets[:, 2:3].detach().clone(),
                        'dim': bbox_targets[:, 3:6].detach().clone(),
                        'rot': bbox_targets[:, 6:].detach().clone(),
                    }
                    query_gt_dict['heatmap'] = F.one_hot(labels.detach().clone(), num_classes=self.num_classes+1)[..., :-1].transpose(1, 2).float()
            else:
                heads = ['heatmap', 'center', 'height', 'dim', 'rot']
                query_gt_dict = {k: None for k in heads}

            second_res_layer = self.prediction_heads[i](query_feat, query_pos, first_res_layer, query_gt_dict)
            res_layer.update(second_res_layer)
            # for next level positional embedding
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
            res_layer['center_mean'] = res_layer['center_mean'] + query_pos.permute(0, 2, 1)
            if 'center_p' in res_layer.keys():
                res_layer['center_p'] = res_layer['center_p'] + query_pos.permute(0, 2, 1)
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)
            

            if not self.fuse_img:
                ret_dicts.append(res_layer)
            #res_layer['gt_dict'] = query_gt_dict

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        if self.fuse_img:
            # positional encoding for image fusion
            img_feat = raw_img_feat.permute(0, 3, 1, 2, 4) # [BS, n_views, C, H, W]
            img_feat_flatten = img_feat.view(batch_size, self.num_views, num_channel, -1)  # [BS, n_views, C, H*W]
            if self.img_feat_pos is None:
                (h, w) = img_inputs.shape[-2], img_inputs.shape[-1]
                img_feat_pos = self.img_feat_pos = self.create_2D_grid(h, w).to(img_feat_flatten.device)
            else:
                img_feat_pos = self.img_feat_pos

            prev_query_feat = query_feat.detach().clone()
            query_feat = torch.zeros_like(query_feat)  # create new container for img query feature
            query_pos_realmetric = query_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]
            query_pos_3d = torch.cat([query_pos_realmetric, res_layer['height']], dim=1).detach().clone()
            if 'vel' in res_layer:
                vel = copy.deepcopy(res_layer['vel'].detach())
            else:
                vel = None
            pred_boxes = self.bbox_coder.decode(
                copy.deepcopy(res_layer['heatmap'].detach()),
                copy.deepcopy(res_layer['rot'].detach()),
                copy.deepcopy(res_layer['dim'].detach()),
                copy.deepcopy(res_layer['center'].detach()),
                copy.deepcopy(res_layer['height'].detach()),
                vel,
            )

            on_the_image_mask = torch.ones([batch_size, self.num_proposals]).to(query_pos_3d.device) * -1

            for sample_idx in range(batch_size if self.fuse_img else 0):
                lidar2img_rt = (img_metas[sample_idx]['lidar2img']).type_as(query_pos_3d)
                img_scale_factor = (
                    query_pos_3d.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                            if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0]))
                img_flip = img_metas[sample_idx]['flip'] if 'flip' in img_metas[sample_idx].keys() else False
                img_crop_offset = (
                    query_pos_3d.new_tensor(img_metas[sample_idx]['img_crop_offset'])
                    if 'img_crop_offset' in img_metas[sample_idx].keys() else 0)
                img_shape = img_metas[sample_idx]['img_shape'][:2]
                img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
                boxes = LiDARInstance3DBoxes(pred_boxes[sample_idx]['bboxes'][:, :7], box_dim=7)
                query_pos_3d_with_corners = torch.cat([query_pos_3d[sample_idx], boxes.corners.permute(2, 0, 1).view(3, -1)], dim=-1)  # [3, num_proposals] + [3, num_proposals*8]
                # transform point clouds back to original coordinate system by reverting the data augmentation
                if batch_size == 1:  # skip during inference to save time
                    points = query_pos_3d_with_corners.T
                else:
                    points = apply_3d_transformation(query_pos_3d_with_corners.T, 'LIDAR', img_metas[sample_idx], reverse=True).detach()
                num_points = points.shape[0]

                for view_idx in range(self.num_views):
                    pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)
                    pts_2d = pts_4d @ lidar2img_rt[view_idx].t()

                    pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
                    pts_2d[:, 0] /= pts_2d[:, 2]
                    pts_2d[:, 1] /= pts_2d[:, 2]

                    # img transformation: scale -> crop -> flip
                    # the image is resized by img_scale_factor
                    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
                    img_coors -= img_crop_offset

                    # grid sample, the valid grid range should be in [-1,1]
                    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

                    if img_flip:
                        # by default we take it as horizontal flip
                        # use img_shape before padding for flip
                        orig_h, orig_w = img_shape
                        coor_x = orig_w - coor_x

                    coor_x, coor_corner_x = coor_x[0:self.num_proposals, :], coor_x[self.num_proposals:, :]
                    coor_y, coor_corner_y = coor_y[0:self.num_proposals, :], coor_y[self.num_proposals:, :]
                    coor_corner_x = coor_corner_x.reshape(self.num_proposals, 8, 1)
                    coor_corner_y = coor_corner_y.reshape(self.num_proposals, 8, 1)
                    coor_corner_xy = torch.cat([coor_corner_x, coor_corner_y], dim=-1)

                    h, w = img_pad_shape
                    on_the_image = (coor_x > 0) * (coor_x < w) * (coor_y > 0) * (coor_y < h)
                    on_the_image = on_the_image.squeeze()
                    # skip the following computation if no object query fall on current image
                    if on_the_image.sum() <= 1:
                        continue
                    on_the_image_mask[sample_idx, on_the_image] = view_idx

                    # add spatial constraint
                    center_ys = (coor_y[on_the_image] / self.out_size_factor_img)
                    center_xs = (coor_x[on_the_image] / self.out_size_factor_img)
                    centers = torch.cat([center_xs, center_ys], dim=-1).int()  # center on the feature map
                    corners = (coor_corner_xy[on_the_image].max(1).values - coor_corner_xy[on_the_image].min(1).values) / self.out_size_factor_img
                    radius = torch.ceil(corners.norm(dim=-1, p=2) / 2).int()  # radius of the minimum circumscribed circle of the wireframe
                    sigma = (radius * 2 + 1) / 6.0
                    distance = (centers[:, None, :] - (img_feat_pos - 0.5)).norm(dim=-1) ** 2
                    gaussian_mask = (-distance / (2 * sigma[:, None] ** 2)).exp()
                    gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
                    attn_mask = gaussian_mask

                    query_feat_view = prev_query_feat[sample_idx, :, on_the_image]
                    query_pos_view = torch.cat([center_xs, center_ys], dim=-1)
                    query_feat_view = self.decoder[self.num_decoder_layers](query_feat_view[None], img_feat_flatten[sample_idx:sample_idx + 1, view_idx], query_pos_view[None], img_feat_pos, attn_mask=attn_mask.log())
                    query_feat[sample_idx, :, on_the_image] = query_feat_view.clone()

            self.on_the_image_mask = (on_the_image_mask != -1)
            res_layer = self.prediction_heads[self.num_decoder_layers](torch.cat([query_feat, prev_query_feat], dim=1))
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
            for key, value in res_layer.items():
                pred_dim = value.shape[1]
                res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)] = first_res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)]
            ret_dicts.append(res_layer)

        if self.initialize_by_heatmap:
            ret_dicts[0]['query_heatmap_score'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
            if self.fuse_img:
                ret_dicts[0]['dense_heatmap'] = dense_heatmap_img
            else:
                ret_dicts[0]['dense_heatmap'] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ['dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score', 'gt_dict']:
                if ret_dicts[0][key] is not None:
                    new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, img_feats, img_metas, gt_bboxes_3d=None, gt_labels_3d=None):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if img_feats is None:
            img_feats = [None]
        res = multi_apply(self.forward_single, feats, img_feats, [img_metas], [gt_bboxes_3d], [gt_labels_3d])
        assert len(res) == 1, "only support one level features."
        return res

    def get_gt_dict(self, gt_bboxes_3d, gt_labels_3d, bev_pos, top_proposals_index):
        #print(len(gt_bboxes_3d))
        res_tuple = []
        for batch_idx in range(len(gt_bboxes_3d)):
            res_tuple.append(self.get_gt_dict_single(gt_bboxes_3d[batch_idx], gt_labels_3d[batch_idx], bev_pos[batch_idx], top_proposals_index[batch_idx]))

        assert len(gt_bboxes_3d) == len(res_tuple)
        gt_dict = {}
        #print(res_tuple)
        for k in res_tuple[0].keys():
            gt_dict[k] = torch.stack([x[k] for x in res_tuple], dim=0)
        return gt_dict

    def get_gt_dict_single(self, gt_bboxes_3d, gt_labels_3d, bev_pos, top_proposals_index):
        device = bev_pos.device
        gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
        reg_map = gt_bboxes_3d.new_zeros(8, feature_map_size[1], feature_map_size[0])
        bev_pos = bev_pos.view(feature_map_size[1], feature_map_size[0], 2)
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']
            if width > 0 and length > 0:
                radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)
                x, y = int(center[0]), int(center[1])
                left, right = min(x, radius), min(feature_map_size[0] - x, radius + 1)
                top, bottom = min(y, radius), min(feature_map_size[1] - y, radius + 1)
                targets = torch.zeros([8, 1, 1]).to(device)
                targets[0] = (gt_bboxes_3d[idx][0] - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                targets[1] = (gt_bboxes_3d[idx][1] - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']
                targets[3] = gt_bboxes_3d[idx][3].log()
                targets[4] = gt_bboxes_3d[idx][4].log()
                targets[5] = gt_bboxes_3d[idx][5].log()
                targets[2] = gt_bboxes_3d[idx][2] + gt_bboxes_3d[idx][5] * 0.5  # bottom center to gravity center
                targets[6] = torch.sin(gt_bboxes_3d[idx][6])
                targets[7] = torch.cos(gt_bboxes_3d[idx][6])
                reg_map[:, y - top:y + bottom, x - left:x + right] = targets.repeat(1, bottom+top, left+right)
                reg_map[:2, y - top:y + bottom, x - left:x + right] -= bev_pos[y - top:y + bottom, x - left:x + right].permute(2, 0, 1)

        heatmap = heatmap.view(self.num_classes, -1)
        reg_map = reg_map.view(8, -1)
        proposal_heatmap = heatmap.gather(index=top_proposals_index[None, :].expand(self.num_classes, -1), dim=1)
        proposal_regmap = reg_map.gather(index=top_proposals_index[None, :].expand(8, -1), dim=1)
        labels = torch.argmax(proposal_heatmap, dim=0)
        neg_mask = (proposal_heatmap.sum(0) == 0)
        labels[neg_mask] = -1
        gt_single_dict = {
            'cls': proposal_heatmap,
            'center': proposal_regmap[:2],
            'height': proposal_regmap[2:3],
            'dim': proposal_regmap[3:6],
            'rot': proposal_regmap[6:8],
            'labels': labels,
            'bbox_weights': torch.max(proposal_heatmap, dim=0).values,
        }
        return gt_single_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                if (key not in ['gt_dict']) and ('mean' not in key) and ('var' not in key):
                    pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, list_of_pred_dict, np.arange(len(gt_labels_3d)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap
        else:
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1)]
            #print(bboxes_tensor_layer)

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, gt_labels_3d, score_layer, self.train_cfg)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, None, gt_labels_3d, self.query_labels[batch_idx])
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        if self.initialize_by_heatmap:
            device = labels.device
            gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device)
            grid_size = torch.tensor(self.train_cfg['grid_size'])
            pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
            voxel_size = torch.tensor(self.train_cfg['voxel_size'])
            feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len]
            heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3]
                length = gt_bboxes_3d[idx][4]
                width = width / voxel_size[0] / self.train_cfg['out_size_factor']
                length = length / voxel_size[1] / self.train_cfg['out_size_factor']
                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)
                    draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)

            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None]

        else:
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        if self.initialize_by_heatmap:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        else:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()

        #print(ious.shape)
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        if self.initialize_by_heatmap:
            # compute heatmap loss
            loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            loss_dict['loss_heatmap'] = loss_heatmap
        # compute loss for each layer
        for idx_layer in range(self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            layer_labels = labels[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            layer_label_weights = label_weights[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]

            layer_score = preds_dict['heatmap'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_heatmap = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls = self.loss_cls(layer_heatmap, layer_labels, layer_label_weights.reshape(-1), avg_factor=max(num_pos, 1))
            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls

            layer_score_q = preds_dict['heatmap_mean'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_heatmap_q = layer_score_q.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls_q = self.loss_cls(layer_heatmap_q, layer_labels, layer_label_weights.reshape(-1), avg_factor=max(num_pos, 1))
            #loss_dict[f'{prefix}_loss_cls_q'] = layer_loss_cls_q

            layer_score_p = preds_dict['heatmap_p'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_heatmap_p = layer_score_p.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls_p = self.loss_cls(layer_heatmap_p, layer_labels, layer_label_weights.reshape(-1), avg_factor=max(num_pos, 1))
            loss_dict[f'{prefix}_loss_cls_p'] = layer_loss_cls_p

            layer_cls_sims = preds_dict['heatmap_sims']
            layer_loss_contrast_cls = (-layer_cls_sims * (layer_cls_sims + 1e-3).log()).sum(dim=1).sum(dim=-1)
            layer_loss_contrast_cls = (layer_loss_contrast_cls * layer_label_weights).sum() / max(num_pos, 1)

            layer_labels = labels[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            layer_label_weights = label_weights[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]

            layer_center = preds_dict['center'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_height = preds_dict['height'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_rot = preds_dict['rot'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_dim = preds_dict['dim'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]

            layer_loss_bbox = self.loss_pbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))
            layer_center = preds_dict['center_p'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_height = preds_dict['height_p'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_rot = preds_dict['rot_p'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_dim = preds_dict['dim_p'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            preds_p = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            layer_loss_pbox = self.loss_pbox(preds_p, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))

            layer_center_sims = preds_dict['center_sims']
            #print(layer_center_sims.shape)
            eps = 1e-3
            layer_loss_contrast_center = (layer_center_sims * (layer_center_sims + eps).log()).sum(dim=1)
            layer_height_sims = preds_dict['height_sims']
            layer_loss_contrast_height = (layer_height_sims * (layer_height_sims + eps).log()).sum(dim=1)
            layer_dim_sims = preds_dict['dim_sims']
            layer_loss_contrast_dim = (layer_dim_sims * (layer_dim_sims + eps).log()).sum(dim=1)
            layer_rot_sims = preds_dict['rot_sims']
            layer_loss_contrast_rot = (layer_rot_sims * (layer_rot_sims + eps).log()).sum(dim=1)
            layer_loss_contrast_reg = torch.cat([layer_loss_contrast_center, layer_loss_contrast_height, layer_loss_contrast_dim, layer_loss_contrast_rot], dim=-1)
            layer_loss_contrast_reg = (-layer_loss_contrast_reg * layer_reg_weights).sum() / max(num_pos, 1)
            
            #loss_dict[f'{prefix}_loss_cls_dist'] = self.loss_dist(preds_dict['cls_attn_map_x'].view(layer_label_weights.shape[0], self.num_classes, -1), preds_dict['cls_attn_map_xy'].view(layer_label_weights.shape[0], self.num_classes, -1), layer_label_weights.unsqueeze(-1), avg_factor=max(num_pos, 1))
            
            
            targets_dict = {
                'center_weights': layer_reg_weights[:, :, :2],
                'height_weights': layer_reg_weights[:, :, 2:3],
                'dim_weights': layer_reg_weights[:, :, 3:6],
                'rot_weights': layer_reg_weights[:, :, 6:8],
            }
            
            if self.loss_bbox_type == 'L1Loss':
                layer_center = preds_dict['center_mean'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                layer_height = preds_dict['height_mean'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                layer_rot = preds_dict['rot_mean'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                layer_dim = preds_dict['dim_mean'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                preds_q = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
                if 'vel' in preds_dict.keys():
                    layer_vel = preds_dict['vel'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                    preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]

                layer_loss_qbox = self.loss_bbox(preds_q, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))
            elif self.loss_bbox_type == 'GaussianLoss':
                layer_center = preds_dict['center_mean'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                layer_height = preds_dict['height_mean'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                layer_rot = preds_dict['rot_mean'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                layer_dim = preds_dict['dim_mean'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                preds_q = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]

                layer_center_var = preds_dict['center_var'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                layer_height_var = preds_dict['height_var'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                layer_rot_var = preds_dict['rot_var'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                layer_dim_var = preds_dict['dim_var'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                preds_var = torch.cat([layer_center_var, layer_height_var, layer_dim_var, layer_rot_var], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
                layer_loss_qbox = self.loss_bbox(preds_q, preds_var, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))
                '''
                layer_loss_bbox_list = []
                for head in ['center', 'height', 'dim', 'rot']:
                    preds_mean = preds_dict[head+'_mean'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].permute(0, 2, 1)
                    preds_var = preds_dict[head+'_var'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].permute(0, 2, 1)
                    preds_p = preds_dict[head+'_p'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].permute(0, 2, 1)
                    #print(preds_mean.shape, preds_var.shape, layer_bbox_targets.shape)
                    layer_loss_pbox_list.append(self.loss_pbox(preds_p, targets_dict[head+'_targets'], targets_dict[head+'_weights'], avg_factor=max(num_pos, 1)))
                    layer_loss_bbox_list.append(self.loss_bbox(preds_mean, preds_var, targets_dict[head+'_targets'], targets_dict[head+'_weights'], avg_factor=max(num_pos, 1)))
                layer_loss_bbox = layer_loss_bbox_list[0] + layer_loss_bbox_list[2] + layer_loss_bbox_list[3] + layer_loss_bbox_list[1]
                layer_loss_pbox = layer_loss_pbox_list[0] + layer_loss_pbox_list[2] + layer_loss_pbox_list[3] + layer_loss_pbox_list[1]
                '''
            
            for head in ['center', 'height', 'dim', 'rot']:
                #print(preds_dict[head+'_attn_map_x'].shape, targets_dict[head+'_weights'].shape)
                
                loss_dict[f'{prefix}_{head}_loss_dist'] = self.loss_dist(preds_dict[head+'_attn_map_x'], preds_dict[head+'_attn_map_xy'], targets_dict[head+'_weights'][..., 0:1], avg_factor=max(num_pos, 1))
            


            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            loss_dict[f'{prefix}_pbox'] = layer_loss_pbox
            loss_dict[f'{prefix}_loss_qbox'] = layer_loss_qbox
            loss_dict[f'{prefix}_loss_cls_p'] = layer_loss_cls_p
            loss_dict[f'{prefix}_loss_cls_q'] = layer_loss_cls_q
            #print(layer_loss_contrast_cls * 0.01, layer_loss_contrast_reg * 0.01)
            loss_dict[f'{prefix}_loss_contrast_cls'] = layer_loss_contrast_cls * 0.2
            loss_dict[f'{prefix}_loss_contrast_reg'] = layer_loss_contrast_reg * 0.2
            #print(loss_dict)
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict[f'matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][..., -self.num_proposals:].sigmoid()
            if 'heatmap_mean' in preds_dict[0]:
               batch_score = torch.sqrt(batch_score * preds_dict[0]['heatmap_mean'][..., -self.num_proposals:].sigmoid())
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0]['query_heatmap_score'] * one_hot

            batch_center = preds_dict[0]['center_mean'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height_mean'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim_mean'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot_mean'][..., -self.num_proposals:]

            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]
            

            temp = self.bbox_coder.decode(batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel, filter=True)

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                    dict(num_class=1, class_names=['pedestrian'], indices=[8], radius=0.175),
                    dict(num_class=1, class_names=['traffic_cone'], indices=[9], radius=0.175),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(num_class=1, class_names=['Car'], indices=[0], radius=0.7),
                    dict(num_class=1, class_names=['Pedestrian'], indices=[1], radius=0.7),
                    dict(num_class=1, class_names=['Cyclist'], indices=[2], radius=0.7),
                ]
            elif self.test_cfg['dataset'] == 'once':
                self.tasks = [
                    dict(num_class=3, class_names=[], indices=[0,1,2], radius=-1),
                    dict(num_class=1, class_names=['Cyclist'], indices=[3], radius=0.85),
                    dict(num_class=1, class_names=['Pedestrian'], indices=[4], radius=0.175),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                ## adopt circle nms for different categories
                if self.test_cfg['nms_type'] != None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat([boxes3d[task_mask][:, :2], scores[:, None][task_mask]], dim=1)
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask])
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1

        res = [[
            img_metas[0]['box_type_3d'](rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
            rets[0][0]['scores'],
            rets[0][0]['labels'].int()
        ]]


        return res
