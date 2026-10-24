"""Multi-head attention (self and cross) with RoPE, GQA and KV-cache decoding.

Provides the flash-attention style attention stack used by
:class:`~.AttentionBlock.AttentionBlock`, the fourth block of the eScreen backbone
(Supplementary Methods §2.1)::

    MHA                      # projection + RoPE + residual-free attention core
      |-- SelfAttention      # torch.nn.functional.scaled_dot_product_attention
      |-- FlashSelfAttention # CUDA flash-attn kernels (requires flash-attn)
      |-- CrossAttention / FlashCrossAttention
      `-- RotaryEmbedding    # from .RotaryEmbedding

plus :class:`ParallelMHA` (tensor-parallel variant) and the ALiBi helper
:func:`get_alibi_slopes`.

What eScreen actually runs
--------------------------
``AttentionBlock`` constructs ``MHA`` with ``embed_dim = hidden_size`` (128
distilled / 256 full), ``num_heads = 8``, ``num_heads_kv = num_heads // 1``,
``rotary_emb_dim = head_dim``, ``rotary_emb_base = 1e6``, ``causal = True``,
``qkv_proj_bias = False``, ``out_proj_bias = True`` and
``use_flash_attn = False``.  That selects the pure-PyTorch
:class:`SelfAttention` core (SDPA) and the fused ``Wqkv`` path, i.e. *none* of the
flash-attn code in this module is executed.  The ``causal=True`` flag is what makes
this block lower-triangular -- see the note in ``AttentionBlock``.

Shapes: input ``[B, L, D]``, packed projection ``[B, L, 3H, D/H]``, attention
scores ``[B, H, L, L]``, output ``[B, L, D]`` (``L = 512``, bfloat16).

Adapted from the StripedHyena2 reference implementation (itself based on the
flash-attention ``layers/rotary.py`` + Megatron-style ``MHA``).
"""

import torch
import torch.nn as nn
from .RotaryEmbedding import RotaryEmbedding

import math
from functools import partial
from einops import rearrange, repeat



try:
    from .ops import (
        local_flash_attn_kvpacked_func,
        local_flash_attn_qkvpacked_func,
        local_flash_attn_varlen_kvpacked_func,
        local_flash_attn_varlen_qkvpacked_func,
        local_flash_attn_with_kvcache,
    )
except ImportError:
    local_flash_attn_varlen_qkvpacked_func, local_flash_attn_varlen_kvpacked_func = (
        None,
        None,
    )
    local_flash_attn_qkvpacked_func, local_flash_attn_kvpacked_func = None, None
    local_flash_attn_with_kvcache = None

FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None





def get_dim_for_local_rank(dim: int, world_size: int, local_rank: int, multiple_of: int = 1) -> int:
    """Get the dim for the local rank derived from splitting dim on world_size processes.

    The split may not be even across the world_size processes.
    # The remainder is handed out one unit at a time (`multiple_of` granularity), so
    # the first `mod` ranks get one extra unit -- this keeps the per-rank sizes
    # summing exactly to `dim` even when `dim` is not divisible by `world_size`.
    """
    multiple = dim // multiple_of
    div = multiple // world_size
    mod = multiple % world_size
    local_multiple = div + int(local_rank < mod)
    return local_multiple * multiple_of





# simple wrapper for flash-attn RoPE with linear scaling:
class LinearlyScaledRotaryEmbedding(RotaryEmbedding):
    """RoPE with linear position scaling (NTK-free "position interpolation").

    An angle at position ``t`` is computed as if the position were
    ``t / scaling_factor``, which compresses ``scaling_factor`` times more context
    into the same angular range.  eScreen does not use it (it is only reachable via
    ``AttentionBlock(use_interpolated_rotary_pos_emb=True)``, a branch that is dead
    for other reasons too).

    Parameters
    ----------
    dim : int
        Width of the rotated subspace (the head dimension in eScreen).
    scaling_factor : float, optional
        Position divisor; 1.0 reproduces plain RoPE.
    base : float, optional
        RoPE frequency base (1e6 from ``AttentionBlock``).
    interleaved : bool, optional
        Channel pairing convention; False is GPT-NeoX style.
    scale_base : float or None, optional
        If given, enables XPos scaling.
    pos_idx_in_fp32 : bool, optional
        Build the position index in float32 (recommended; see the parent class).
    device : torch.device, optional
        Device for the frequency buffer.
    """
    def __init__(
        self,
        dim: int,
        scaling_factor: float = 1.0,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """Store the scaling factor; the frequency tables are inherited unchanged."""
        super().__init__(
            dim=dim,
            base=base,
            interleaved=interleaved,
            scale_base=scale_base,
            pos_idx_in_fp32=pos_idx_in_fp32,
            device=device,
        )
        self._linear_scaling_factor = scaling_factor

    # adpated from: https://github.com/Dao-AILab/flash-attention/blob/43ceab630bc6c27712428da5a33fc9cb5c369d91/flash_attn/layers/rotary.py#L368
    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        """Rebuild the cos/sin tables with positions divided by the scaling factor.

        Parameters
        ----------
        seqlen : int
            Number of positions to cache.
        device : torch.device, optional
            Device of the tables.
        dtype : torch.dtype, optional
            Storage dtype of the tables (matches the q/k dtype).

        Returns
        -------
        None
            Writes ``_cos_cached`` / ``_sin_cached`` (and the XPos variants).

        Notes
        -----
        This is a copy of the parent method with the single extra line
        ``t = t / self._linear_scaling_factor``; the cache-invalidation condition
        and the float32 rationale are identical.  Keeping the copy means the parent
        class stays bit-identical to upstream.
        """
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # linear scaling:
                t = t / self._linear_scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                # linear scaling:
                t = t / self._linear_scaling_factor
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)


# swap out RoPE of existing mha:
def swap_mha_rope(
    mha,
    new_rope: torch.nn.Module = LinearlyScaledRotaryEmbedding,
    kwargs_new_rope: dict = None,
):
    """Replace an existing ``MHA`` instance's RoPE module in place.

    Parameters
    ----------
    mha : MHA
        Module whose ``rotary_emb`` attribute is swapped.
    new_rope : type, optional
        Replacement class (default :class:`LinearlyScaledRotaryEmbedding`).
    kwargs_new_rope : dict, optional
        Extra constructor arguments for the new class (default
        ``{"scaling_factor": 1.0}``).

    Returns
    -------
    MHA
        The same object, mutated.

    Notes
    -----
    The old module's hyper-parameters are read back from the instance (not from the
    config) so that the replacement is guaranteed to rotate the same subspace.  The
    deletion-then-assignment is what makes PyTorch re-register the submodule; a
    plain attribute assignment would leave the old module registered under the same
    name only if it were an ``nn.Module`` attribute set via ``__setattr__`` -- which
    it is, so the explicit ``del`` is belt-and-braces.
    """
    # determine mha dtype and device:
    dtype = mha.Wq.weight.dtype if mha.cross_attn else mha.Wqkv.weight.dtype
    device = mha.Wq.weight.device if mha.cross_attn else mha.Wqkv.weight.device
    # determine RoPE settings:
    kwargs_old_rope = dict(
        dim=mha.rotary_emb.dim,
        base=mha.rotary_emb.base,
        interleaved=mha.rotary_emb.interleaved,
        scale_base=mha.rotary_emb.scale_base,
        pos_idx_in_fp32=mha.rotary_emb.pos_idx_in_fp32,
        device=mha.rotary_emb.inv_freq.device,
    )
    # delete old RoPE:
    del mha.rotary_emb
    # create new RoPE:
    kwargs_new_rope = kwargs_new_rope or {"scaling_factor": 1.0}
    scaled_rope = new_rope(**kwargs_new_rope, **kwargs_old_rope).to(dtype)
    # attach new RoPE to mha:
    mha.rotary_emb = scaled_rope
    # make new sure RoPE is correctly registered:
    assert isinstance(mha.rotary_emb, new_rope)
    return mha







# From https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
def get_alibi_slopes(nheads):
    """ALiBi attention slopes (Press et al.), one per head.

    Parameters
    ----------
    nheads : int
        Number of attention heads; need not be a power of two.

    Returns
    -------
    list of float
        Slopes in decreasing order of magnitude, so head 0 has the strongest
        recency bias.

    Notes
    -----
    For a power of two the slopes are ``2^(-8/n) ** i``; otherwise the largest power
    of two ``n' <= n`` contributes its slopes and the remaining ``n - n'`` heads take
    every second slope of a ``2n'`` table, which reproduces the published ALiBi
    recipe.  eScreen does not use ALiBi (``use_alibi=False``).
    """
    def get_slopes_power_of_2(nheads):
        """Geometric slope sequence for a power-of-two head count.

        Parameters
        ----------
        nheads : int
            Power-of-two number of heads.

        Returns
        -------
        list of float
            ``start * start**i`` for ``i = 0 .. nheads-1`` with
            ``start = 2**(-8/nheads)``, i.e. the published ALiBi slopes.
        """
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
        )


    
    
    
    
    
class FlashSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        layer_number,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    ):
        """Store the attention switches; the flash kernel is validated at call time.

        Parameters
        ----------
        layer_number : int
            Layer index (stored only; the kernels do not use it).
        causal : bool, optional
            Default mask direction.
        softmax_scale : float or None, optional
            ``None`` means the flash kernel uses ``1/sqrt(head_dim)``.
        attention_dropout : float, optional
            Dropout probability (0.0 in eScreen).
        window_size : tuple, optional
            ``(left, right)`` sliding window; ``(-1, -1)`` means unbounded.
        alibi_slopes : torch.Tensor, optional
            Registered as a non-persistent buffer when given.
        deterministic : bool, optional
            Use the deterministic (slower) backward kernel.

        Notes
        -----
        The two asserts at the top fail at *construction* time when the flash-attn
        extension could not be imported, which is why ``MHA`` only instantiates this
        class when ``use_flash_attn=True``.
        """
        super().__init__()
        assert local_flash_attn_varlen_qkvpacked_func is not None, "FlashAttention is not installed"
        assert local_flash_attn_qkvpacked_func is not None, "FlashAttention is not installed"
        self.layer_number = layer_number
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
        self.window_size = window_size
        self.deterministic = deterministic

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes.to(torch.float32)
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            return local_flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )
        else:
            y = local_flash_attn_qkvpacked_func(
                qkv,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )

            return y


class FlashCrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        alibi_slopes=None,
        window_size=(-1, -1),
        deterministic=False,
    ):
        """Store the cross-attention switches (same fields as FlashSelfAttention)."""
        super().__init__()
        assert local_flash_attn_varlen_kvpacked_func is not None, "FlashAttention is not installed"
        assert local_flash_attn_kvpacked_func is not None, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
        self.window_size = window_size
        self.deterministic = deterministic

    def forward(
        self,
        q,
        kv,
        causal=None,
        cu_seqlens=None,
        max_seqlen=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
            max_seqlen: int. Maximum sequence length in the batch of q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
            max_seqlen_k: int. Maximum sequence length in the batch of k and v.
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes.to(torch.float32)
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            assert cu_seqlens_k is not None
            assert cu_seqlens_k.dtype == torch.int32
            assert max_seqlen_k is not None
            assert isinstance(max_seqlen_k, int)
            return local_flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_seqlens,
                cu_seqlens_k,
                max_seqlen,
                max_seqlen_k,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )
        else:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            seqlen_k = kv.shape[1]
            assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
            return local_flash_attn_kvpacked_func(
                q,
                kv,
                self.drop.p if self.training else 0.0,
                causal=causal,
                softmax_scale=self.softmax_scale,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )


class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        """SDPA-based attention core: the path eScreen actually executes.

        Parameters
        ----------
        causal : bool, optional
            Default causal masking (overridable per call).
        softmax_scale : float or None, optional
            Scale applied to ``QK^T`` before softmax; ``None`` means
            ``1/sqrt(head_dim)``, which is what SDPA applies by default.
        attention_dropout : float, optional
            Dropout probability (0.0 in eScreen, so no dropout at all).
        """
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        # The line below scales q by (softmax_scale * sqrt(d)).  SDPA already divides
        # QK^T by sqrt(d) internally, so this pre-multiplication is exactly what turns
        # the built-in 1/sqrt(d) into an arbitrary user-supplied softmax_scale; with
        # the default (softmax_scale = 1/sqrt(d)) the factor telescopes to 1 and the
        # line is a no-op.  Doing it here rather than in the matmul avoids a second
        # pass over the scores.
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, D)
        q = q.permute(0, 2, 1, 3)  # (B, H, T, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        batch_size, num_heads, seqlen, d = q.shape

        scale = self.softmax_scale if self.softmax_scale is not None else 1.0 / math.sqrt(d)
        q = q * (scale * math.sqrt(d))

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.where(
                repeat(key_padding_mask, "b s -> b t s", t=seqlen),
                0.0,
                -10000.0,
            )

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=(self.causal if causal is None else causal),
        )

        output = output.permute(0, 2, 1, 3)
        return output


class CrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        """Explicit (non-flash) cross-attention: manual softmax over ``QK^T``.

        Used only when ``use_flash_attn=False`` *and* cross-attention is requested;
        eScreen uses self-attention, so this class is not exercised.
        """
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, Sk)
        """
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        causal = self.causal if causal is None else causal
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
        if kv.shape[3] != q.shape[2]:  # MQA/GQA
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen_k),
                -10000.0,
                dtype=scores.dtype,
                device=scores.device,
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # causal mask needs to take into account the difference between seqlen_q and seqlen_k
            row_idx = rearrange(torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1")
            col_idx = torch.arange(seqlen_k, device=kv.device, dtype=torch.long)
            sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
            causal_mask = col_idx > row_idx + sk - seqlen_q
            scores = scores.masked_fill(causal_mask, -10000.0)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense."""
    # Returns (output, input) so that a caller can fuse the residual addition with
    # the backward pass of the Linear; see the return_residual flag of MHA.  The
    # tuple is why ParallelGatedMLP has grab_first_if_tuple().

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``(Linear(input), input)``.

        Parameters
        ----------
        input : torch.Tensor
            ``[..., in_features]``.

        Returns
        -------
        tuple
            ``(output, input)`` -- the second element is the residual, passed
            through untouched so the caller can add it (or fuse the addition with
            the backward pass).
        """
        return super().forward(input), input


def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # The cache is allocated lazily on first use with the maximum shapes from the
    # InferenceParams object, then written in place at the current offset.  The
    # returned view stops at the current sequence end, so the attention core only
    # sees valid keys.
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_seqlen,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        kv_cache = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention"""
    # Structure of forward(): one fused QKV projection (or separate Q and KV for
    # cross-attention / GQA), optional depth-wise conv, optional RoPE, then the
    # attention core, then the output projection.  The residual is *not* added here
    # -- AttentionBlock adds it -- so this module is a pure mixing operator.

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        cross_attn=False,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        fused_bias_fc=False,
        use_flash_attn=False,
        return_residual=False,
        checkpointing=False,
        device=None,
        dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        # Only the fused-QKV path (num_heads_kv == num_heads, no cross-attention)
        # is used by eScreen; the GQA/cross-attention branches exist for the
        # upstream model family.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing
        if use_alibi:
            assert use_flash_attn, "ALiBi code path requires flash_attn"
            alibi_slopes = torch.tensor(get_alibi_slopes(num_heads), device=device)
        else:
            alibi_slopes = None
        if window_size != (-1, -1):
            assert use_flash_attn, "Local (sliding window) attention code path requires flash_attn"

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert self.num_heads % self.num_heads_kv == 0, "num_heads must be divisible by num_heads_kv"
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        if self.rotary_emb_dim > 0:
            assert not cross_attn, "MHA with rotary embedding does not support cross-attention yet"
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_resid_cls = LinearResidual if not fused_bias_fc else partial(FusedDense, return_residual=True)
        wqkv_cls = linear_cls if not self.return_residual else linear_resid_cls
        inner_attn_cls = (
            partial(
                FlashSelfAttention,
                layer_number=self.layer_idx,
                alibi_slopes=alibi_slopes,
                window_size=window_size,
            )
            if use_flash_attn
            else SelfAttention
        )
        inner_cross_attn_cls = (
            partial(FlashCrossAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else CrossAttention
        )
        if not self.cross_attn:
            self.Wqkv = wqkv_cls(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        else:
            self.Wq = linear_cls(embed_dim, embed_dim, bias=qkv_proj_bias, **factory_kwargs)
            self.Wkv = wqkv_cls(embed_dim, kv_dim, bias=qkv_proj_bias, **factory_kwargs)
        if self.dwconv:
            if self.num_heads_kv == self.num_heads:
                self.dwconv_qkv = nn.Conv1d(qkv_dim, qkv_dim, kernel_size=3, padding=2, groups=qkv_dim)
            else:
                self.dwconv_q = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2, groups=embed_dim)
                self.dwconv_kv = nn.Conv1d(kv_dim, kv_dim, kernel_size=3, padding=2, groups=kv_dim)
        self.inner_attn = inner_attn_cls(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=dropout,
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.out_proj = linear_cls(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """Allocate an empty KV cache for inference.

        Parameters
        ----------
        batch_size : int
            Maximum batch size.
        max_seqlen : int
            Maximum sequence length to cache.
        dtype : torch.dtype, optional
            Cache dtype; defaults to the output projection's dtype.

        Returns
        -------
        torch.Tensor
            ``[batch_size, max_seqlen, 2, num_heads_kv, head_dim]``, uninitialised.
        """
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        # ``layer_idx`` must have been passed to the constructor: it is the key under
        # which this layer's cache lives, so decoding with layer_idx=None fails the
        # assert rather than silently sharing one cache between layers.
        assert not self.dwconv, "Generation does not support dwconv yet"
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        # Fused path for cached decoding: the flash kernel applies RoPE to the new
        # q/k, appends them to the cache and attends in one launch.  Requires
        # seqlen_offset > 0 (i.e. at least one token already cached) and a rotary
        # width divisible by 16 -- both checked by the caller before branching here.
        assert inference_params is not None and inference_params.seqlen_offset > 0
        assert self.use_flash_attn
        if self.rotary_emb_dim > 0:
            assert self.rotary_emb.scale is None, "This code path does not support xPos"
            self.rotary_emb._update_cos_sin_cache(inference_params.max_seqlen, device=q.device, dtype=q.dtype)
            rotary_cos, rotary_sin = (
                self.rotary_emb._cos_cached,
                self.rotary_emb._sin_cached,
            )
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
        context = local_flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.inner_cross_attn.softmax_scale,
            causal=self.inner_cross_attn.causal,
            rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
            alibi_slopes=alibi_slopes,
        )
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        # Either update the cache by hand and run the ordinary attention core (first
        # step, no flash, or no kvcache kernel available), or let the flash kvcache
        # kernel do the append + attention in one call.
        if inference_params.seqlen_offset == 0 or local_flash_attn_with_kvcache is None or not self.use_flash_attn:
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
            return local_flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.inner_cross_attn.softmax_scale,
                causal=self.inner_cross_attn.causal,
                alibi_slopes=alibi_slopes,
            )

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        # Branch summary for the version eScreen runs (self-attention,
        # num_heads_kv == num_heads): Wqkv -> optional dwconv -> RoPE -> SelfAttention
        # -> out_proj.  The ``inference_params is None`` case is the only one taken
        # during training and evaluation, and it is the branch whose attention core
        # receives the constructor's causal=True.
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
            assert not self.dwconv
            assert self.rotary_emb_dim == 0
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn
        if inference_params is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None
            assert not self.dwconv

        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
            if self.use_flash_attn
            else {"key_padding_mask": key_padding_mask, **kwargs}
        )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        batch, seqlen = x.shape[:2]
        if not self.cross_attn and self.num_heads_kv == self.num_heads:
            assert x_kv is None and mixer_subset is None
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            if self.dwconv:
                qkv = rearrange(
                    self.dwconv_qkv(rearrange(qkv, "b s d -> b d s"))[..., :-2],
                    "b d s -> b s d",
                ).contiguous()
            qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen)
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_attn(qkv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
                else:
                    context = self._update_kvcache_attention(qkv[:, :, 0], qkv[:, :, 1:], inference_params)
            else:
                context = self._apply_rotary_update_kvcache_attention(qkv[:, :, 0], qkv[:, :, 1:], inference_params)
        else:
            if self.cross_attn:
                if not self.return_residual:
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
                    kv = self.Wkv(x_kv if x_kv is not None else x)
                else:
                    if x_kv is not None:
                        kv, x_kv = self.Wkv(x_kv)
                    else:
                        kv, x = self.Wkv(x)
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
            else:
                assert self.num_heads_kv != self.num_heads
                if not self.return_residual:
                    qkv = self.Wqkv(x)
                else:
                    qkv, x = self.Wqkv(x)
                q = qkv[..., : self.num_heads * self.head_dim]
                kv = qkv[..., self.num_heads * self.head_dim :]
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
            kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
            if self.dwconv:
                q = rearrange(
                    self.dwconv_q(rearrange(q, "b s d -> b d s"))[..., :-2],
                    "b d s -> b s d",
                ).contiguous()
                kv = rearrange(
                    self.dwconv_kv(rearrange(kv, "b s d -> b d s"))[..., :-2],
                    "b d s -> b s d",
                ).contiguous()
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen)
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_cross_attn(q, kv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_cross_attn, q, kv, **kwargs)
                else:
                    context = self._update_kvcache_attention(q, kv, inference_params)
            else:
                context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)


class ParallelMHA(nn.Module):
    """Multi-head self-attention and cross-attention"""
    # NOTE: unusable in this checkout.  The tensor-parallel layers are hard-set to
    # ``FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None`` at
    # the top of the module (the fused_dense package is not vendored), so
    # ParallelMHA.__init__ always raises ImportError.  Nothing in eScreen
    # instantiates it -- model parallelism is 1 everywhere.

    def __init__(
        self,
        embed_dim,
        num_heads,
        process_group,
        num_heads_kv=None,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        use_flash_attn=False,
        checkpointing=False,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ) -> None:
        """Set up the tensor-parallel projections.

        Parameters
        ----------
        embed_dim : int
            Full (unsharded) model width ``D``.
        num_heads : int
            Total number of heads across all ranks.
        process_group : torch.distributed.ProcessGroup
            Group over which QKV is column- and the output row-sharded.
        num_heads_kv : int, optional
            K/V heads for GQA; defaults to ``num_heads``.
        qkv_proj_bias, out_proj_bias : bool, optional
            Biases for the sharded projections.
        dropout, softmax_scale, causal, layer_idx, rotary_emb_dim,
        rotary_emb_base, rotary_emb_scale_base, rotary_emb_interleaved,
        use_alibi, window_size : optional
            Same meaning as in :class:`MHA`, applied per rank.
        use_flash_attn : bool, optional
            Selects the flash cores (required for ALiBi / sliding window).
        checkpointing : bool, optional
            Recompute the attention core in the backward pass.
        sequence_parallel : bool, optional
            Shard the sequence axis instead of the head axis.
        device, dtype : optional
            Passed to the sharded layers.

        Notes
        -----
        ``multiple_of`` on the column-parallel layer keeps every rank's QKV shard a
        multiple of ``head_dim * (num_heads/num_heads_kv + 2)``, i.e. each rank gets a
        whole number of Q/K/V head groups.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpointing = checkpointing
        self.process_group = process_group
        self.world_size = process_group.size()
        self.local_rank = torch.distributed.get_rank(process_group)

        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert self.num_heads % self.num_heads_kv == 0, "num_heads must be divisible by num_heads_kv"

        self.num_heads_per_rank = get_dim_for_local_rank(self.num_heads, self.world_size, self.local_rank)
        self.num_heads_kv_per_rank = get_dim_for_local_rank(self.num_heads_kv, self.world_size, self.local_rank)
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)

        if use_alibi:
            assert use_flash_attn, "ALiBi code path requires flash_attn"
            num_heads_local = math.ceil(self.num_heads / self.world_size)
            alibi_slopes = torch.tensor(
                get_alibi_slopes(num_heads)[
                    self.local_rank * num_heads_local : (self.local_rank + 1) * num_heads_local
                ],
                device=device,
            )
        else:
            alibi_slopes = None
        if window_size != (-1, -1):
            assert use_flash_attn, "Local (sliding window) attention code path requires flash_attn"

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.Wqkv = ColumnParallelLinear(
            embed_dim,
            qkv_dim,
            process_group,
            bias=qkv_proj_bias,
            sequence_parallel=sequence_parallel,
            multiple_of=self.head_dim * (self.num_heads // self.num_heads_kv + 2),
            **factory_kwargs,
        )
        inner_attn_cls = (
            partial(FlashSelfAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else SelfAttention
        )
        inner_cross_attn_cls = (
            partial(FlashCrossAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else CrossAttention
        )
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            process_group,
            bias=out_proj_bias,
            sequence_parallel=sequence_parallel,
            multiple_of=self.head_dim,
            **factory_kwargs,
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """Allocate the per-rank KV cache: ``[B, L, 2, heads_kv_per_rank, head_dim]``."""
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv_per_rank,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        # Same bookkeeping as MHA._update_kv_cache; only the head count differs
        # (per-rank shard instead of the full head count).
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        # Same fused decode path as MHA: RoPE + cache append + attention in one
        # kernel, available only when tokens are already cached and the rotary width
        # is a multiple of 16.
        assert inference_params is not None and inference_params.seqlen_offset > 0
        assert self.use_flash_attn
        if self.rotary_emb_dim > 0:
            assert self.rotary_emb.scale is None, "This code path does not support xPos"
            self.rotary_emb._update_cos_sin_cache(inference_params.max_seqlen, device=q.device, dtype=q.dtype)
            rotary_cos, rotary_sin = (
                self.rotary_emb._cos_cached,
                self.rotary_emb._sin_cached,
            )
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
        context = local_flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.inner_cross_attn.softmax_scale,
            causal=self.inner_cross_attn.causal,
            rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
            alibi_slopes=alibi_slopes,
        )
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        # Cache write + attention, or the fused flash kvcache kernel when a cache
        # already exists and flash-attn is in use.
        if inference_params.seqlen_offset == 0 or not self.use_flash_attn:
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
            context = local_flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.inner_cross_attn.softmax_scale,
                causal=self.inner_cross_attn.causal,
                alibi_slopes=alibi_slopes,
            )
            return context

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)
        if seqlen is not None:
            qkv = rearrange(qkv, "(b s) ... -> b s ...", s=seqlen)
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        if self.num_heads_kv == self.num_heads:
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen)
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_attn(qkv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
                else:
                    context = self._update_kvcache_attention(qkv[:, :, 0], qkv[:, :, 1:], inference_params)
            else:
                context = self._apply_rotary_update_kvcache_attention(qkv[:, :, 0], qkv[:, :, 1:], inference_params)
        else:
            q = rearrange(
                qkv[..., : self.num_heads_per_rank * self.head_dim],
                "... (h d) -> ... h d",
                d=self.head_dim,
            )
            kv = rearrange(
                qkv[..., self.num_heads_per_rank * self.head_dim :],
                "... (two hkv d) -> ... two hkv d",
                two=2,
                d=self.head_dim,
            )
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen)
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_cross_attn(q, kv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_cross_attn, q, kv, **kwargs)
                else:
                    context = self._update_kvcache_attention(q, kv, inference_params)
            else:
                context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        context = rearrange(context, "b s h d -> b s (h d)")
        if seqlen is not None:
            context = rearrange(context, "b s d -> (b s) d")
        out = self.out_proj(context)
        return out
