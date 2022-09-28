import copy
import warnings
from xml.dom import NotFoundErr
import numpy as np
import torch
import mmcv
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn
import os
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
            key_pos_embed = None

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

            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            out_layer = build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True)
            self.__setattr__(head+'_out', out_layer)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head+'_out').bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv1d):
                        kaiming_init(m)
                for m in self.__getattr__(head+'_out').modules():
                    if isinstance(m, nn.Conv1d):
                        kaiming_init(m)


    def forward(self, x, postfix=None):
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
            y = self.__getattr__(head)(x)
            if postfix is not None:
                if (postfix != 'c') and (head == 'center'):
                    ret_dict['corner_'+postfix] = self.__getattr__(head+'_out')(y)
                else:
                    ret_dict[head+'_'+postfix] = self.__getattr__(head+'_out')(y)
            else:
                ret_dict[head] = self.__getattr__(head+'_out')(y)

        return ret_dict

class GatedCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True):
        super(GatedCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = float(self.head_dim) ** -0.5
        self.bias = bias

        self.in_proj_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.gate_layer = nn.Sequential(nn.Linear(8, 4 * embed_dim), nn.Sigmoid())
        self._reset_parameters()
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if self.bias:
                    constant_(m.bias, 0.)

    def forward(self, query, key, gate):
        B, N, C = query.shape
        q = self.in_proj_q(query)
        k = self.in_proj_k(key)
        v = self.in_proj_v(key)
        g = self.gate_layer(gate)

        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, 4, self.num_heads, self.head_dim)
        v = k.view(B, N, 4, self.num_heads, self.head_dim)
        g = k.view(B, N, 4, self.num_heads, self.head_dim)
        attn_map = torch.einsum('bnhc,bnmhc->bnmh', q, k)
        attn_map = torch.softmax(attn_map * self.scale, dim=-2)
        out = torch.einsum('bnmh,bnmhc->bnhc', attn_map, v*g)
        out = out.view(B, N, C)
        out = self.out_proj(out)
        return out

class GatedCrossTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_channel, dropout=0.0, activation='relu', bias=True):
        super().__init__()
        self.attn = GatedCrossAttention(embed_dim, num_heads, bias=bias)
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

    def forward(self, x, y, gate):
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 3, 1)
        feat = self.attn(x, y, gate)
        x = x + self.dropout2(feat)
        x = self.norm2(x)
        features2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(features2)
        x = self.norm3(x)
        x = x.permute(0, 2, 1)
        return x

class GatedMerge(nn.Module):
    def __init__(self, embed_dim, ffn_channel, dropout, activation='relu'):
        self.proj_layer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.gate_layer = nn.Sequential(nn.Linear(8, 4 * embed_dim), nn.Sigmoid())
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

    def forward(self, x, y, rel_pos, gate):
        x = x.permute(0, 2, 3, 1)
        y = y.unsqueeze(-1) + rel_pos
        y = y.permute(0, 2, 3, 1)
        feat = self.proj_layer(y)
        g = self.gate_layer(gate)
        g = g.view(*(y.shape))
        x = x + self.dropout2(feat * g)
        x = self.norm2(x)
        features2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(features2)
        x = self.norm3(x)
        x = x.permute(0, 3, 1, 2)
        return x

def rotation_2d(points, rots):
    rot_sin = torch.sin(rots)
    rot_cos = torch.cos(rots)
    rot_mat = torch.stack([torch.stack([rot_cos, -rot_sin]), torch.stack([rot_sin, rot_cos])])
    if len(points.shape) == 2:
        points = torch.einsum('bi,ijb->bj', [points, rot_mat])
    elif len(points.shape) == 3:
        points = torch.einsum('bin,ijbn->bjn', [points, rot_mat])
    else:
        points = torch.einsum('bain,ijbn->bajn', [points, rot_mat])
    return points

class GeometryLayer(nn.Module):
    def __init__(self,
                 hidden_channels,
                 num_classes,
                 geometry_cfg,
                 bbox_coder):
        super(GeometryLayer, self).__init__()

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.train_cfg = geometry_cfg
        if self.train_cfg is not None:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))

        self.rel_pos_emb = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))

        self.class_encoding = nn.Sequential(
            nn.Conv2d(num_classes, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))
        
        num_heads = geometry_cfg['num_heads']
        ffn_channel = geometry_cfg['ffn_channel']
        dropout = geometry_cfg['dropout']
        activation = geometry_cfg['activation']

        self.corner_to_center_attn = GatedCrossTransformer(hidden_channels, num_heads, ffn_channel, dropout, activation, bias=True)
        self.center_to_corner_layer = GatedMerge(hidden_channels, ffn_channel, dropout, activation)

        self.pc_range = geometry_cfg['pc_range']
        self.voxel_size = geometry_cfg['voxel_size']
        self.out_size_factor = geometry_cfg['out_size_factor']

    def center_to_corner(self, centers, dims, rots):
        corners = torch.from_numpy(np.stack(np.unravel_index(np.arange(4), [2]*2), dim=1))
        corners = corners - 0.5
        corners = corners.view(1, 4, 2, 1).type_as(dims) * dims[:, :2, :].exp().unsqueeze(1)
        rot = torch.atan2(rots[:, 0, :], rots[:, 1, :])
        corners = self.rotation_2d(corners, rot)
        corners[:, :, 0, :] = (corners[:, :, 0, :]) / self.voxel_size[0] / self.out_size_factor
        corners[:, :, 1, :] = (corners[:, :, 1, :]) / self.voxel_size[1] / self.out_size_factor
        corners = corners + centers.unsqueeze(1)
        return corners.permute(0, 2, 3, 1)
    
    def hungarian_matching(self, preds_dicts, batch_size, postfix):
        match_dict_list = []
        for i in range(batch_size):
            preds_dict = {}
            for k in preds_dicts.keys():
                preds_dict[k] = preds_dicts[k][i:i+1].detach().clone()
            match_dict = self.hungarian_matching_single(preds_dict, postfix)
            match_dict_list.append(match_dict)
        
        match_dicts = {}
        for k in match_dict_list[0].keys():
            match_dicts[k] = torch.stack([x[k] for x in match_dict_list], dim=0)
        return match_dicts

    def hungarian_matching_single(self, preds_dict, postfix):

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap_c'])
        center = copy.deepcopy(preds_dict['center_c'])
        height = copy.deepcopy(preds_dict['height_c'])
        dim = copy.deepcopy(preds_dict['dim_c'])
        rot = copy.deepcopy(preds_dict['rot_c'])
        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, None)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']

        match_dict = {}

        for p in postfix:
            gt_score = copy.deepcopy(preds_dict['heatmap_'+p])
            gt_center = copy.deepcopy(preds_dict['center_'+p])
            gt_height = copy.deepcopy(preds_dict['height_'+p])
            gt_dim = copy.deepcopy(preds_dict['dim_'+p])
            gt_rot = copy.deepcopy(preds_dict['rot_'+p])
            gt_boxes_dict = self.bbox_coder.decode(gt_score, gt_rot, gt_dim, gt_center, gt_height, None)  # decode the prediction to real world metric bbox
            gt_bboxes_tensor = gt_boxes_dict[0]['bboxes']
            assigned_gt_inds, max_overlaps = self.bbox_assigner.assign(bboxes_tensor, gt_bboxes_tensor, gt_score[0], score[0], self.train_cfg)
            match_dict['assigned_inds_'+p] = assigned_gt_inds.long()
            match_dict['overlaps_'+p] = max_overlaps

        return match_dict

    def with_embed(self, x, x_pos=None, x_class=None):
        if x_pos is not None:
            if x_class is not None:
                return x + x_pos + x_class
            else:
                return x + x_pos
        else:
            if x_class is not None:
                return x + x_class
            else:
                return x 

    def forward(self, feature_dict, preds_dict, postfix, seperate=False):
        B, C, N = feature_dict['query_feat_c'].shape
        match_dict = self.hungarian_matching(preds_dict, B, postfix)
        if not seperate:
            feat_ld = feature_dict['query_feat_ld'].gather(index=match_dict['assigned_inds_ld'].unsqueeze(1).expand(B, C, N), dim=-1)
            feat_lt = feature_dict['query_feat_lt'].gather(index=match_dict['assigned_inds_lt'].unsqueeze(1).expand(B, C, N), dim=-1)
            feat_rd = feature_dict['query_feat_rd'].gather(index=match_dict['assigned_inds_rd'].unsqueeze(1).expand(B, C, N), dim=-1)
            feat_rt = feature_dict['query_feat_rt'].gather(index=match_dict['assigned_inds_rt'].unsqueeze(1).expand(B, C, N), dim=-1)
            feat_corner = torch.stack([feat_ld, feat_lt, feat_rd, feat_rt], dim=-1)

            center_ld = preds_dict['center_ld'].gather(index=match_dict['assigned_inds_ld'].unsqueeze(1).expand(B, C, N), dim=-1)
            center_lt = preds_dict['center_lt'].gather(index=match_dict['assigned_inds_lt'].unsqueeze(1).expand(B, C, N), dim=-1)
            center_rd = preds_dict['center_rd'].gather(index=match_dict['assigned_inds_rd'].unsqueeze(1).expand(B, C, N), dim=-1)
            center_rt = preds_dict['center_rt'].gather(index=match_dict['assigned_inds_rt'].unsqueeze(1).expand(B, C, N), dim=-1)
            center_corner = torch.stack([center_ld, center_lt, center_rd, center_rt], dim=-1)
            pos_embed_corner = self.pos_emb(center_corner)

            score_ld = preds_dict['heatmap_ld'].gather(index=match_dict['assigned_inds_ld'].unsqueeze(1).expand(B, C, N), dim=-1)
            score_lt = preds_dict['heatmap_lt'].gather(index=match_dict['assigned_inds_lt'].unsqueeze(1).expand(B, C, N), dim=-1)
            score_rd = preds_dict['heatmap_rd'].gather(index=match_dict['assigned_inds_rd'].unsqueeze(1).expand(B, C, N), dim=-1)
            score_rt = preds_dict['heatmap_rt'].gather(index=match_dict['assigned_inds_rt'].unsqueeze(1).expand(B, C, N), dim=-1)
            score_corner = torch.stack([score_ld, score_lt, score_rd, score_rt], dim=-1)
            class_embed_corner = self.class_encoding(score_corner)

            feat_c = feature_dict['query_feat_c']
            center_c = preds_dict['center_c']
            score_c = preds_dict['heatmap_c']
            pos_embed_c = self.pos_emb(center_c.unsqueeze(-1)).squeeze(-1)
            class_embed_c = self.class_encoding(score_c.unsqueeze(-1)).squeeze(-1)

            r = torch.linalg.norm(center_c, dim=1)
            theta = torch.atan2(center_c[:, 0, :], center_c[:, 1, :])
            feat_gate = torch.cat([
                r.unsqueeze(-1),
                theta.unsqueeze(-1),
                preds_dict['rot_c'].transpose(1, 2),
                match_dict['overlaps_ld'].unsqueeze(-1),
                match_dict['overlaps_lt'].unsqueeze(-1),
                match_dict['overlaps_rd'].unsqueeze(-1),
                match_dict['overlaps_rt'].unsqueeze(-1)
            ], dim=-1)

            feat_c = self.corner_to_center_attn(self.with_embed(feat_c, pos_embed_c, class_embed_c), self.with_embed(feat_corner, pos_embed_corner, class_embed_corner), feat_gate)

            corner_ld = preds_dict['corner_ld'].gather(index=match_dict['assigned_inds_ld'].unsqueeze(1).expand(B, C, N), dim=-1)
            corner_lt = preds_dict['corner_lt'].gather(index=match_dict['assigned_inds_lt'].unsqueeze(1).expand(B, C, N), dim=-1)
            corner_rd = preds_dict['corner_rd'].gather(index=match_dict['assigned_inds_rd'].unsqueeze(1).expand(B, C, N), dim=-1)
            corner_rt = preds_dict['corner_rt'].gather(index=match_dict['assigned_inds_rt'].unsqueeze(1).expand(B, C, N), dim=-1)
            corner = torch.stack([corner_ld, corner_lt, corner_rd, corner_rt], dim=-1)
            corner_c = self.center_to_corner(center_c, preds_dict['dim_c'], preds_dict['rot_c'])
            rel_pos_embed = self.rel_pos_emb(corner - corner_c)
            feat_corner = self.center_to_corner_layer(feat_corner, feat_c, rel_pos_embed, feat_gate)
            feat_ld = feat_corner.new_zeros(B, C, N)
            torch.scatter(feat_ld, dim=-1, index=match_dict['assigned_inds_ld'].unsqueeze(1).expand(B, C, N), src=feat_corner[..., 0])
            torch.scatter(feat_lt, dim=-1, index=match_dict['assigned_inds_lt'].unsqueeze(1).expand(B, C, N), src=feat_corner[..., 1])
            torch.scatter(feat_rd, dim=-1, index=match_dict['assigned_inds_rd'].unsqueeze(1).expand(B, C, N), src=feat_corner[..., 2])
            torch.scatter(feat_rt, dim=-1, index=match_dict['assigned_inds_rt'].unsqueeze(1).expand(B, C, N), src=feat_corner[..., 3])

        out_dict = {'feat_c': feat_c, 'feat_ld': feat_ld, 'feat_lt': feat_lt, 'feat_rd': feat_rd, 'feat_rt': feat_rt}
        return out_dict


@HEADS.register_module()
class CTransFusionHead(nn.Module):
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
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 geometry_cfg=None,
                 ):
        super(CTransFusionHead, self).__init__()

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

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        self.postfix = ['ld', 'lt', 'rd', 'rt']

        if self.initialize_by_heatmap:
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
            for p in self.postfix:
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
                self.__setattr__('heatmap_head_' + p, nn.Sequential(*layers))
                #self.heatmap_head = nn.Sequential(*layers)
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
        self.prediction_heads = nn.ModuleList()
        self.prediction_heads_corner = nn.ModuleList()
        #self.second_prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))
            self.prediction_heads_corner.append(FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))
        #self.inner_object_layer = GeometryLayer(hidden_channel, self.num_classes, geometry_cfg, bbox_coder)

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

    def corner_to_center(self, corners, dims, rots, postfix):
        if postfix == 'ld':
            center = [0.5, 0.5]
        elif postfix == 'lt':
            center = [0.5, -0.5]
        elif postfix == 'rd':
            center = [-0.5, 0.5]
        else:
            center = [-0.5, -0.5]
        center = dims.new_tensor(center)
        center = center.view(1, 2, 1) * dims[:, :2, :].exp()
        rot = torch.atan2(rots[:, 0, :], rots[:, 1, :])
        center = rotation_2d(center, rot)
        center[:, 0, :] = (center[:, 0, :]) / self.test_cfg['voxel_size'][0] / self.test_cfg['out_size_factor']
        center[:, 1, :] = (center[:, 1, :]) / self.test_cfg['voxel_size'][1] / self.test_cfg['out_size_factor']
        center = center + corners
        return center

    def center_to_corner(self, boxes, postfix):
        if postfix == 'ld':
            corner = [-0.5, -0.5]
        elif postfix == 'lt':
            corner = [-0.5, 0.5]
        elif postfix == 'rd':
            corner = [0.5, -0.5]
        else:
            corner = [0.5, 0.5]
        corner = boxes.new_tensor(corner)
        corners = corner.view(1, 2).type_as(boxes) * boxes[:, 3:5]
        corners = rotation_2d(corners, boxes[:, 6])
        corners = corners + boxes[:, :2]
        new_boxes = torch.cat([corners, boxes[:, 2:]], dim=1)
        return new_boxes

    def forward_single(self, inputs, img_inputs, img_metas):
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
        query_dicts = dict()
        if self.initialize_by_heatmap:
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
            query_dicts['dense_heatmap_c'] = dense_heatmap

            # top #num_proposals among all classes
            top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
            top_proposals_class = top_proposals // heatmap.shape[-1]
            top_proposals_index = top_proposals % heatmap.shape[-1]
            query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
            query_dicts['query_labels_c'] = top_proposals_class
            query_dicts['query_heatmap_score_c'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]

            # add category embedding
            one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
            query_cat_encoding = self.class_encoding(one_hot.float())
            query_dicts['query_feat_c'] = query_feat + query_cat_encoding

            query_dicts['query_pos_c'] = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
            for p in self.postfix:
                dense_heatmap = self.__getattr__('heatmap_head_' + p)(lidar_feat)
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
                query_dicts['dense_heatmap_'+p] = dense_heatmap

                # top #num_proposals among all classes
                top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
                top_proposals_class = top_proposals // heatmap.shape[-1]
                top_proposals_index = top_proposals % heatmap.shape[-1]
                query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
                query_dicts['query_labels_' + p] = top_proposals_class
                query_dicts['query_heatmap_score_'+p] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]

                # add category embedding
                one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
                query_cat_encoding = self.class_encoding(one_hot.float())
                query_dicts['query_feat_'+p] = query_feat + query_cat_encoding

                query_dicts['query_pos_'+p] = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)

        else:
            print(self.initialize_by_heatmap)
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
            preds_dict = {}
            query_dicts['query_feat_c'] = self.decoder[i](query_dicts['query_feat_c'], lidar_feat_flatten, query_dicts['query_pos_c'], bev_pos)
            res_layer = self.prediction_heads[i](query_dicts['query_feat_c'], postfix='c')
            res_layer['center_c'] = res_layer['center_c'] + query_dicts['query_pos_c'].permute(0, 2, 1)

            for p in self.postfix:
                query_dicts['query_feat_'+p] = self.decoder[i](query_dicts['query_feat_'+p], lidar_feat_flatten, query_dicts['query_pos_'+p], bev_pos)
                res_layer_ = self.prediction_heads_corner[i](query_dicts['query_feat_'+p], postfix=p)
                res_layer_['corner_'+p] = res_layer_['corner_'+p] + query_dicts['query_pos_'+p].permute(0, 2, 1)
                #print(query_dicts['query_pos_'+p].permute(0, 2, 1))
                res_layer_['center_'+p] = self.corner_to_center(res_layer_['corner_'+p], res_layer_['dim_'+p], res_layer_['rot_'+p], postfix=p)
                #print(res_layer_['center_'+p][:, :20], res_layer_['corner_'+p][:, :20])
                #print(res_layer_['dim_'+p].exp(), torch.atan2(res_layer_['rot_'+p][:,0,:], res_layer_['rot_'+p][:,1,:]))
                res_layer.update(res_layer_)

            '''
            for k in res_layer.keys():
                preds_dict[k] = res_layer[k].detach().clone()

            # Prediction
            feature_dict = self.inner_object_layer(query_dicts, preds_dict, self.postfix)
            second_res_layer = self.second_prediction_heads[i](feature_dict['feat_c'], postfix='c')
            second_res_layer['center_c'] = second_res_layer['center_c'] + query_dicts['query_pos_c'].permute(0, 2, 1)
            for p in self.postfix:
                second_res_layer_ = self.sencond_prediction_heads[i](feature_dict['feat_'+p], postfix=p)
                second_res_layer_['corner_'+p] = second_res_layer['corner_'+p] + query_dicts['query_pos_'+p].permute(0, 2, 1)
                second_res_layer_['center_'+p] = self.corner_to_center(second_res_layer_['corner_'+p], second_res_layer_['dim_'+p], second_res_layer_['rot_'+p], postfix=p)
                second_res_layer.update(second_res_layer_)
            '''
            
            if not self.fuse_img:
                ret_dicts.append(res_layer)
            
            first_res_layer = res_layer

            # for next level positional embedding
            query_pos = res_layer['center_c'].detach().clone().permute(0, 2, 1)

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
            for p in self.postfix:
                ret_dicts[0]['query_heatmap_score_'+p] = query_dicts['query_heatmap_score_'+p]
                if self.fuse_img:
                    ret_dicts[0]['dense_heatmap_'+p] = dense_heatmap_img
                else:
                    ret_dicts[0]['dense_heatmap_'+p] = query_dicts['dense_heatmap_'+p]
            ret_dicts[0]['query_heatmap_score_c'] = query_dicts['query_heatmap_score_c']
            if self.fuse_img:
                ret_dicts[0]['dense_heatmap'] = dense_heatmap_img
            else:
                ret_dicts[0]['dense_heatmap_c'] = query_dicts['dense_heatmap_c']

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if ('dense_heatmap' not in key) and ('query_heatmap_score' not in key):
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, img_feats, img_metas):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if img_feats is None:
            img_feats = [None]
        res = multi_apply(self.forward_single, feats, img_feats, [img_metas])
        assert len(res) == 1, "only support one level features."
        return res

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
        list_of_gt_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]

            out_dict = self.get_targets_single(gt_bboxes_3d[batch_idx], gt_labels_3d[batch_idx], pred_dict, batch_idx)
            list_of_gt_dict.append(out_dict)

        assert len(gt_bboxes_3d) == len(list_of_gt_dict)

        res_dict = {}
        for k in list_of_gt_dict[0].keys():
            if 'num_pos' in k:
                res_dict[k] = np.sum([x[k] for x in list_of_gt_dict])
            elif 'mean_iou' in k:
                res_dict[k] = np.mean([x[k] for x in list_of_gt_dict])
            else:
                res_dict[k] = torch.cat([x[k] for x in list_of_gt_dict], dim=0)

        
        return res_dict

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
        out_dict = {}
        # center

        num_proposals = preds_dict['center_c'].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap_c'].detach())
        center = copy.deepcopy(preds_dict['center_c'].detach())
        height = copy.deepcopy(preds_dict['height_c'].detach())
        dim = copy.deepcopy(preds_dict['dim_c'].detach())
        rot = copy.deepcopy(preds_dict['rot_c'].detach())
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
            #print(bboxes_tensor_layer.shape, gt_bboxes_tensor.shape)

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

        out_dict['labels_c'] = labels[None]
        out_dict['label_weights_c'] = label_weights[None]
        out_dict['bbox_targets_c'] = bbox_targets[None]
        out_dict['bbox_weights_c'] = bbox_weights[None]
        out_dict['ious_c'] = ious[None]
        out_dict['num_pos_c'] = int(pos_inds.shape[0])
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
            out_dict['heatmap_c'] = heatmap[None]
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        out_dict['mean_iou_c'] = float(mean_iou)

        for p in self.postfix:
            num_proposals = preds_dict['center_'+p].shape[-1]

            # get pred boxes, carefully ! donot change the network outputs
            score = copy.deepcopy(preds_dict['heatmap_'+p].detach())
            center = copy.deepcopy(preds_dict['center_'+p].detach())
            height = copy.deepcopy(preds_dict['height_'+p].detach())
            dim = copy.deepcopy(preds_dict['dim_'+p].detach())
            rot = copy.deepcopy(preds_dict['rot_'+p].detach())
            if 'vel' in preds_dict.keys():
                vel = copy.deepcopy(preds_dict['vel'].detach())
            else:
                vel = None

            boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)  # decode the prediction to real world metric bbox
            bboxes_tensor = boxes_dict[0]['bboxes']
            gt_corners_tensor = self.center_to_corner(gt_bboxes_tensor, p)
            
            #print(gt_corners_tensor, gt_bboxes_tensor)
            # each layer should do label assign seperately.
            if self.auxiliary:
                num_layer = self.num_decoder_layers
            else:
                num_layer = 1

            assign_result_list = []
            for idx_layer in range(num_layer):
                bboxes_tensor_layer = bboxes_tensor[self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1), :]
                score_layer = score[..., self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1)]

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
            sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bboxes_tensor, gt_corners_tensor)
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
            
            out_dict['labels_'+p] = labels[None]
            out_dict['label_weights_'+p] = label_weights[None]
            out_dict['bbox_targets_'+p] = bbox_targets[None]
            out_dict['bbox_weights_'+p] = bbox_weights[None]
            out_dict['ious_'+p] = ious[None]
            out_dict['num_pos_'+p] = int(pos_inds.shape[0])

            # # compute dense heatmap targets
            if self.initialize_by_heatmap:
                device = labels.device
                grid_size = torch.tensor(self.train_cfg['grid_size'])
                pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
                voxel_size = torch.tensor(self.train_cfg['voxel_size'])
                feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len]
                heatmap = gt_corners_tensor.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
                for idx in range(len(gt_corners_tensor)):
                    width = gt_corners_tensor[idx][3]
                    length = gt_corners_tensor[idx][4]
                    width = width / voxel_size[0] / self.train_cfg['out_size_factor']
                    length = length / voxel_size[1] / self.train_cfg['out_size_factor']
                    if width > 0 and length > 0:
                        radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                        radius = max(self.train_cfg['min_radius'], int(radius))
                        x, y = gt_corners_tensor[idx][0], gt_corners_tensor[idx][1]

                        coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                        coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']

                        center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                        center_int = center.to(torch.int32)
                        draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)
                out_dict['heatmap_'+p] = heatmap[None]

            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            out_dict['mean_iou_'+p] = float(mean_iou)
        return out_dict

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
        with torch.no_grad():
            res_dict = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        
        '''
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        '''
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        for p in (['c'] + self.postfix):
            if self.initialize_by_heatmap:
                # compute heatmap loss
                #print(preds_dict['dense_heatmap_'+p].shape, res_dict['heatmap_'+p].shape)
                loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap_'+p]), res_dict['heatmap_'+p], avg_factor=max(res_dict['heatmap_'+p].eq(1).float().sum().item(), 1))
                loss_dict['loss_heatmap_'+p] = loss_heatmap

            # compute loss for each layer


            layer_labels = res_dict['labels_'+p].reshape(-1)
            layer_label_weights = res_dict['label_weights_'+p].reshape(-1)
            layer_score = preds_dict['heatmap_'+p]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls = self.loss_cls(layer_cls_score, layer_labels, layer_label_weights, avg_factor=max(res_dict['num_pos_'+p], 1))

            if p == 'c':
                layer_center = preds_dict['center_'+p]
            else:
                layer_center = preds_dict['corner_'+p]
            layer_height = preds_dict['height_'+p]
            layer_rot = preds_dict['rot_'+p]
            layer_dim = preds_dict['dim_'+p]
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]

            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = res_dict['bbox_weights_'+p]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
            layer_bbox_targets = res_dict['bbox_targets_'+p]
            layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(res_dict['num_pos_'+p], 1))

            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            loss_dict[f'loss_cls_'+p] = layer_loss_cls
            loss_dict[f'loss_bbox_'+p] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

            loss_dict[f'matched_ious_'+p] = layer_loss_cls.new_tensor(res_dict['mean_iou_'+p])

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
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0]['query_heatmap_score'] * one_hot

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]

            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]
            
            if self.test_cfg.get('store_feature', False):
                batch_feature = preds_dict[0]['features'][..., -self.num_proposals:].transpose(1,2)
            if self.test_cfg.get('store_box_feature', False):
                batch_bev_feat = preds_dict[0]['bev_feature']

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
                    if self.test_cfg.get('store_feature', False):
                        ret['features'] = batch_feature[i][keep_mask]
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    if self.test_cfg.get('store_feature', False):
                        ret['features'] = batch_feature[i]
                if self.test_cfg.get('store_box_feature', False):
                    x = img_metas[0]['box_type_3d'](ret['bboxes'], box_dim=ret['bboxes'].shape[-1])
                    corners = x.corners
                    gravity_center = x.gravity_center
                    s1 = (corners[:, [0,2,4,6], :2] * 2 + gravity_center[:, :2].unsqueeze(1)) / 3
                    s2 = [s1[:,0]+s1[:,1], s1[:,0]+s1[:,2], s1[:,1]+s1[:,3], s1[:,2]+s1[:,3]]
                    s2 = torch.stack(s2, dim=1) / 2
                    s = torch.cat([gravity_center[:, :2].unsqueeze(1), s1, s2], dim=1)
                    s = (s - s.new_tensor(self.test_cfg['pc_range'])) / s.new_tensor(self.test_cfg['voxel_size']) / s.new_tensor(self.test_cfg['grid_size'])[:2]
                    s = s * 2 - 1
                    s_feat = F.grid_sample(batch_bev_feat[i:i+1], s.unsqueeze(0))
                    ret['box_features'] = s_feat.squeeze(0).permute(1, 2, 0).contiguous().view(corners.shape[0], 9*batch_bev_feat.shape[1])
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1
        if self.test_cfg.get('store_feature', False):
            res = [[
                img_metas[0]['box_type_3d'](rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
                rets[0][0]['scores'],
                rets[0][0]['labels'].int(),
                rets[0][0]['features'],
            ]]
        else:
            res = [[
                img_metas[0]['box_type_3d'](rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
                rets[0][0]['scores'],
                rets[0][0]['labels'].int()
            ]]
        if self.test_cfg.get('store_box_feature', False):
            res[0].append(rets[0][0]['box_features'])

        return res
