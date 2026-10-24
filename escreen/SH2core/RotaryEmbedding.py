"""Rotary position embeddings (RoPE) for the eScreen attention block.

Provides the RoPE autograd wrappers and the :class:`RotaryEmbedding` module used
by :class:`~.MHA.MHA`.  In eScreen, ``AttentionBlock`` builds its attention with
``rotary_emb_dim = hidden_size // num_attention_heads`` (the full head width) and
``rotary_emb_base = 1e6``, so every Q/K channel is rotated and no channel is left
un-rotated.

What RoPE does
--------------
For a head vector ``x`` of width ``d`` split into ``d/2`` channel pairs, position
``t`` maps to the angle ``theta_i * t`` with ``theta_i = base^(-2i/d)``.  Each pair
is rotated by that angle::

    [x_{2i}  ]   [ cos(t theta_i)  -sin(t theta_i) ] [x_{2i}  ]
    [x_{2i+1}] = [ sin(t theta_i)   cos(t theta_i) ] [x_{2i+1}]

Because ``R(a) R(b)^T = R(a - b)``, the dot product of a query at position ``t``
with a key at position ``s`` depends only on ``t - s``: the position information
enters the attention logits as a relative offset, and no positional vector is
added to the embeddings.

Implementation notes
--------------------
* All rotations are executed by the Triton kernel in :mod:`ops`
  (``ops.apply_rotary``); the functions here only prepare cos/sin, dispatch, and
  provide the backward pass, which re-runs the kernel with ``conjugate=True``
  because rotation matrices are orthogonal.
* Two channel-pairing conventions are supported: ``interleaved=False`` pairs
  channel ``i`` with ``i + d/2`` (GPT-NeoX style, what eScreen uses) and
  ``interleaved=True`` pairs ``2i`` with ``2i+1`` (GPT-J style).
* Tensors are rotated **in place** wherever the kernel allows it; the wrappers
  therefore mutate their inputs, which is why callers pass freshly projected Q/K/V
  and never a view of the residual stream.

Adapted from the StripedHyena2 / flash-attention implementation.
"""

import torch
from einops import rearrange, repeat
from typing import Optional, Tuple, Union
from .ops import apply_rotary




def rotate_half(x, interleaved=False):
    """Rotate the paired channels of the last dimension by +90 degrees.

    Parameters
    ----------
    x : torch.Tensor
        ``[..., rotary_dim]``; the last axis holds the paired channels.
    interleaved : bool, optional
        ``False`` (default, GPT-NeoX): pair ``x[i]`` with ``x[i + d/2]``.
        ``True`` (GPT-J): pair the adjacent ``x[2i]`` / ``x[2i+1]``.

    Returns
    -------
    torch.Tensor
        ``[-x2, x1]`` with the same shape and dtype as ``x`` -- the vector rotated
        by +90 degrees inside every 2-D subspace.

    Notes
    -----
    Only used by :func:`apply_rotary_emb_torch`, i.e. by the pure-PyTorch
    reference implementation; the path eScreen runs is the Triton kernel in
    ``ops.rotary_kernel``, which performs the same rotation without materialising
    this intermediate.  The two conventions differ merely in how channels are
    paired, so mixing them yields a plausible but wrong rotation.
    """
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    # Pure-PyTorch reference implementation (not used by the model; it is the
    # oracle for ops.apply_rotary).  The two ``repeat`` patterns encode the two
    # pairing conventions: "(2 d)" duplicates each angle for the even/odd slot of a
    # half-split pair, "(d 2)" duplicates it for adjacent pairs.  The trailing
    # ``x[..., ro_dim:]`` keeps channels beyond rotary_dim unrotated, because the
    # rotary width may be smaller than the head width (not the case in eScreen,
    # where rotary_dim == head_dim).
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class ApplyRotaryEmb(torch.autograd.Function):
    """Autograd wrapper around the Triton RoPE kernel for a single tensor.

    ``forward`` rotates ``x``; ``backward`` re-applies the same kernel with
    ``conjugate=True``.  That is exact rather than an approximation: a rotation
    matrix ``R`` satisfies ``dL/dx = R^T dL/dy = R(-theta) dL/dy``, so the
    transposed Jacobian is just the rotation by the negative angle.

    Tensors are ``[B, L, H, D]`` (or ``[total, H, D]`` with ``cu_seqlens``), cos/sin
    are ``[seqlen_ro, rotary_dim/2]``.
    """
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        """Rotate ``x`` and record the parameters that backward needs.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, L, H, D]``, or ``[total, H, D]`` when ``cu_seqlens`` is given.
        cos, sin : torch.Tensor
            ``[seqlen_ro, rotary_dim/2]``, same dtype as ``x`` (asserted in
            ``ops.apply_rotary``).
        interleaved : bool, optional
            Pairing convention (see :func:`rotate_half`).
        inplace : bool, optional
            Overwrite ``x`` instead of writing to a new tensor.
        seqlen_offsets : int or torch.Tensor, optional
            Position offset(s); a tensor of shape ``[B]`` is used with a KV cache.
        cu_seqlens : torch.Tensor, optional
            Cumulative lengths for variable-length batching.
        max_seqlen : int, optional
            Required when ``cu_seqlens`` is given.

        Returns
        -------
        torch.Tensor
            The rotated tensor (``x`` itself when ``inplace``).
        """
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        """Back-propagate through the rotation.

        Parameters
        ----------
        do : torch.Tensor
            Gradient w.r.t. the forward output, same shape as ``x``.

        Returns
        -------
        tuple
            ``(dx, None, None, None, None, None, None, None)`` -- one gradient for
            each forward argument, only ``x`` being differentiable.

        Notes
        -----
        The ``do.clone()`` below is a workaround for a Triton 2.0
        "invalid device context" error when the kernel runs in place on a
        non-contiguous gradient; it is kept as in the upstream code.
        """
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    # Public entry point: thin dispatch to the autograd Function above.  Because
    # the Function rotates in place, the returned tensor may alias ``x`` -- callers
    # must not reuse the input afterwards.
    return ApplyRotaryEmb.apply(x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen)


# For backward compatibility
apply_rotary_emb_func = apply_rotary_emb


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    """Autograd wrapper that rotates Q and K inside a packed QKV tensor in place.

    The packed tensor is either ``[B, L, 3, H, D]`` or, for grouped-query
    attention, ``[B, L, H_q + 2 H_kv, D]`` (then ``num_heads_q`` is required).
    Only the query and key slices are rotated; the value slice is untouched.

    A single kernel launch is used when ``cos_k``/``sin_k`` are not supplied and
    the tensor is contiguous: the ``(3, H)`` axes are then merged into one
    ``(3H)`` axis, which is the same memory as the separate slices.

    Returns
    -------
    torch.Tensor
        The same (mutated) tensor.
    """
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        num_heads_q: Union[int] = None,
    ):
        """Rotate the Q and K slices of ``qkv`` in place.

        Parameters
        ----------
        qkv : torch.Tensor
            ``[B, L, 3, H, D]`` or ``[B, L, H_q + 2 H_kv, D]``.
        cos, sin : torch.Tensor
            ``[seqlen_ro, rotary_dim/2]`` used for the query (and for the key when
            no separate key tables are given).
        cos_k, sin_k : torch.Tensor, optional
            Separate tables for the key, needed for XPos-style scaling.
        interleaved : bool, optional
            Pairing convention.
        seqlen_offsets : int or torch.Tensor, optional
            Position offset(s).
        num_heads_q : int, optional
            Required when ``qkv`` is 4-D (MQA/GQA layout).

        Returns
        -------
        torch.Tensor
            ``qkv``, rotated in place.

        Notes
        -----
        ``ctx.save_for_backward`` is called in both branches below; repeated calls
        append to the context rather than replacing, so the second call is
        redundant but harmless.
        """
        if cos_k is None and sin_k is None and qkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            if qkv.dim() == 5:
                batch, seqlen, three, nheads, headdim = qkv.shape
                assert three == 3
                # qk = rearrange(qkv[:, :, :2], "b s t h d -> b s (t h) d")
                qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            else:
                assert qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                qk = qkv[:, :, : num_heads_q + num_heads_k]
            apply_rotary(
                qk,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=interleaved,
                inplace=True,
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            if qkv.dim() == 5:
                q, k = qkv[:, :, 0], qkv[:, :, 1]
            else:
                assert qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                q, k = (
                    qkv[:, :, :num_heads_q],
                    qkv[:, :, num_heads_q : num_heads_q + num_heads_k],
                )
            apply_rotary(q, cos, sin, seqlen_offsets, interleaved=interleaved, inplace=True)
            apply_rotary(k, cos_k, sin_k, seqlen_offsets, interleaved=interleaved, inplace=True)
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.num_heads_q = num_heads_q
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        """Rotate the Q/K gradient slices back (conjugate rotation).

        Parameters
        ----------
        dqkv : torch.Tensor
            Gradient w.r.t. the packed tensor.

        Returns
        -------
        tuple
            ``(dqkv, None, None, None, None, None, None, None)``.
        """
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k = ctx.saved_tensors
        if cos_k is None and sin_k is None and dqkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need dqkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            if dqkv.dim() == 5:
                dqk = rearrange(dqkv[:, :, :2], "b s t h d -> b s (t h) d")
            else:
                assert dqkv.dim() == 4
                assert ctx.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - ctx.num_heads_q) // 2
                assert dqkv.shape[2] == ctx.num_heads_q + 2 * num_heads_k
                dqk = dqkv[:, :, : ctx.num_heads_q + num_heads_k]
            apply_rotary(
                dqk,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            if dqkv.dim() == 5:
                dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]
            else:
                assert dqkv.dim() == 4
                assert ctx.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - ctx.num_heads_q) // 2
                assert dqkv.shape[2] == ctx.num_heads_q + 2 * num_heads_k
                dq = dqkv[:, :, : ctx.num_heads_q]
                dk = dqkv[:, :, ctx.num_heads_q : ctx.num_heads_q + num_heads_k]
            apply_rotary(
                dq,
                cos,
                sin,
                seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
            apply_rotary(
                dk,
                cos_k,
                sin_k,
                seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
        return dqkv, None, None, None, None, None, None, None


def apply_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    num_heads_q: Optional[int] = None,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim).
            If qkv has shape (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
    """
    # As with the single-tensor wrapper, this mutates ``qkv`` in place and returns
    # the same object, so the caller's projection buffer is overwritten.  eScreen
    # is fine with that: MHA applies RoPE to the fresh ``Wqkv`` output and never
    # needs the un-rotated Q/K again.
    return ApplyRotaryEmbQKV_.apply(qkv, cos, sin, cos_k, sin_k, interleaved, seqlen_offsets, num_heads_q)


class ApplyRotaryEmbKV_(torch.autograd.Function):
    """Autograd wrapper that rotates only the key slice of a packed KV tensor.

    ``kv`` is ``[B, L, 2, H, D]`` (key at index 0, value at index 1).  Only the key
    is rotated, in place; the value is left untouched.  Used for cross-attention
    and for KV-cache updates, where the query is rotated separately by
    :func:`apply_rotary_emb`.
    """
    @staticmethod
    def forward(
        ctx,
        kv,
        cos,
        sin,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
    ):
        """Rotate the key slice of ``kv`` in place.

        Parameters
        ----------
        kv : torch.Tensor
            ``[B, L, 2, H, D]``.
        cos, sin : torch.Tensor
            ``[seqlen_ro, rotary_dim/2]``.
        interleaved : bool, optional
            Pairing convention.
        seqlen_offsets : int or torch.Tensor, optional
            Position offset(s).

        Returns
        -------
        torch.Tensor
            ``kv``, key rotated in place.
        """
        batch, seqlen, two, nheads, headdim = kv.shape
        assert two == 2
        k = kv[:, :, 0]
        apply_rotary(
            k,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=interleaved,
            inplace=True,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        return kv

    @staticmethod
    def backward(ctx, dkv):
        """Rotate the key gradient back (conjugate rotation).

        Parameters
        ----------
        dkv : torch.Tensor
            Gradient w.r.t. ``[B, L, 2, H, D]``.

        Returns
        -------
        tuple
            ``(dkv, None, None, None, None)``.
        """
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors
        apply_rotary(
            dkv[:, :, 0],
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
        )
        return dkv, None, None, None, None


apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply
# NOTE: the alias above is immediately shadowed by the ``def`` below (same name),
# so ``ApplyRotaryEmbKV_.apply`` is not reachable through this name.  The public
# wrapper is the function, and that is what RotaryEmbedding.forward calls; the
# assignment is dead code kept from upstream.


def apply_rotary_emb_kv_(
    kv,
    cos,
    sin,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
):
    """
    Arguments:
        kv: (batch_size, seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        kv: (batch_size, seqlen, 2, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of K.
    """
    # Public entry point for the KV-only rotation; see ApplyRotaryEmbKV_ above.
    return ApplyRotaryEmbKV_.apply(kv, cos, sin, interleaved, seqlen_offsets)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """
    # eScreen instantiates this from MHA with dim = head_dim, base = 1e6 and
    # interleaved = False (GPT-NeoX pairing over the whole head width).

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        # inv_freq and scale are registered with persistent=False: both are pure
        # functions of `dim` / `base`, so they need not (and should not) be stored in
        # the checkpoint, which keeps state_dict keys unchanged when the base is
        # changed in a config.
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        """Build the inverse-frequency table ``theta_i = base^(-2i/dim)``.

        Parameters
        ----------
        device : torch.device, optional
            Device for the new tensor.

        Returns
        -------
        torch.Tensor
            Shape ``[dim/2]``, dtype float32.

        Notes
        -----
        float32 is deliberate even when the model runs in bfloat16: these values are
        multiplied by position indices up to L, and in bf16 the resulting angles
        would be quantised so coarsely that different positions could collide (see
        the ``pos_idx_in_fp32`` note in :meth:`__init__`).
        """
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        """(Re)build the cached cos/sin tables for ``seqlen`` positions.

        Parameters
        ----------
        seqlen : int
            Number of positions to cover.
        device : torch.device, optional
            Device of the cache.
        dtype : torch.dtype, optional
            dtype the tables are stored in (matches the q/k dtype, bfloat16 in
            eScreen).

        Returns
        -------
        None
            The tables are written to ``self._cos_cached`` / ``self._sin_cached``
            (and ``_cos_k_cached`` / ``_sin_k_cached`` for XPos).

        Notes
        -----
        * The cache is rebuilt only when it grows, changes device/dtype, or when
          switching from inference to training mode; a shorter sequence simply
          reuses the longer table and indexes its first positions.
        * Angles are computed in float32 and only the final cos/sin are cast, so the
          bf16 rounding happens once per table entry instead of once per use.
        * ``torch.outer(t, inv_freq)`` is used instead of ``einsum`` on purpose:
          einsum would down-cast float32 to fp16 under AMP.
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

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) or (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
            if kv is none, else it's just q of shape (batch, seqlen, nheads, headdim).
            If qkv has shape (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        # The table is sized from `max_seqlen` when a per-sample offset tensor is
        # used (KV-cache decoding), otherwise from seqlen + a constant offset --
        # otherwise positions beyond the cached range would read a stale table.
        # The rotation itself then runs at qkv.dtype (bfloat16 in eScreen).
        seqlen = qkv.shape[1]
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
        if kv is None:
            if self.scale is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
        else:
            q = qkv
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            return q, kv