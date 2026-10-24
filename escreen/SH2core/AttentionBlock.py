"""Attention residual block -- the fourth and final block of the eScreen backbone.

Provides :class:`AttentionBlock`, a pre-norm transformer block made of a
multi-head self-attention sub-layer (:class:`~.MHA.MHA`, with rotary position
embeddings) followed by a gated feed-forward sub-layer
(:class:`~.ParallelGatedMLP.ParallelGatedMLP`), each wrapped in its own residual
connection.

Role in eScreen
---------------
``escreen/escreen.py`` (``get_block``) stacks four blocks: three
``ParallelGatedConvBlock`` instances (HyenaSE, HyenaMR, HyenaLI) followed by one
``AttentionBlock`` with ``layer_idx = 3``.  The attention block therefore
supplies the "attention" half of the hybrid convolution/attention design
described in Supplementary Methods §2.1, and is the only block in the model that
mixes information globally through a softmax over sequence positions.  Its output
flows into the same mean-pool over the 512 positions as the Hyena blocks.

eScreen configuration (``get_block(..., block_type='Attention')``):
``hidden_size = d_model`` (128 distilled / 256 full), ``num_attention_heads = 8``,
``proj_groups = 1`` (i.e. plain MHA, no GQA), ``qkv_proj_bias = False``,
``mha_out_proj_bias = True``, ``use_flash_attn = False`` (the pure-PyTorch
``SelfAttention`` path), ``rotary_emb_base = 1e6`` and ``inner_mlp_size = 256``.

Adapted from the StripedHyena2 reference implementation.
"""

import torch
import torch.nn as nn
from .RMSNorm import RMSNorm
from .ParallelGatedMLP import ParallelGatedMLP
from .MHA import MHA

class AttentionBlock(nn.Module):
    """Residual multi-head attention block with a residual gated MLP.

    Forward computation (``RMSNorm`` is pre-norm, i.e. applied to the branch
    input, not to the branch output)::

        u = u + MHA(pre_norm(u))        # attention sub-layer
        u = u + MLP(post_norm(u))       # gated feed-forward sub-layer

    Shapes
    ------
    * input / output: ``[B, L, D]`` with ``D = hidden_size`` and ``L = 512`` in
      eScreen (bfloat16 end to end).
    * inside ``MHA``: ``[B, L, H, D/H]`` per projection, ``H =
      num_attention_heads``; rotary embeddings rotate the full head width
      ``D/H``; attention scores are ``[B, H, L, L]``.

    Parameters
    ----------
    hidden_size : int
        Model width ``D``.
    num_attention_heads : int
        Number of attention heads ``H``; ``D`` must be divisible by ``H``
        (128/8 = 16 and 256/8 = 32 in eScreen).
    layer_idx : int
        Index of this block inside the stack (3 in eScreen).  Only forwarded to
        the MLP and to ``MHA`` (where it indexes the KV cache during decoding).
    use_rms_norm : bool, optional
        Accepted but unused: the two norms are always ``RMSNorm``.  Kept so that
        upstream configs can be passed through unchanged.
    proj_groups : int, optional
        Grouped-query factor: ``num_heads_kv = num_heads // proj_groups``.
        ``1`` means plain MHA (eScreen's value).
    attn_block_dtype, mlp_dtype : torch.dtype, optional
        dtypes for the attention and MLP sub-modules (both bfloat16 in eScreen,
        matching the "all computation in bfloat16" statement of §2.1).
    qkv_proj_bias, mha_out_proj_bias : bool, optional
        Biases for the fused QKV projection and the output projection.
    use_flash_attn : bool, optional
        Selects ``FlashSelfAttention`` (CUDA flash-attn kernels) instead of the
        PyTorch SDPA implementation.  eScreen passes False because the flash
        path needs a specific CUDA toolchain.
    rotary_emb_base : float, optional
        RoPE frequency base; eScreen leaves the default ``1e6``.
    use_interpolated_rotary_pos_emb, rotary_emb_scaling_factor : optional
        Request a linearly scaled RoPE.  NOTE: the helper this calls
        (``swap_mha_rope``) is not imported by this module, so the branch raises
        ``NameError`` if enabled; eScreen keeps it disabled.
    smeared_gqa : bool, optional
        If True, sets ``num_heads_kv = num_heads`` after construction.  See the
        note at that assignment -- it is only consistent with ``proj_groups == 1``.
    **kwargs
        MLP hyper-parameters (``inner_size_multiple_of``, ``mlp_activation``,
        ``evo2_style_activations``, ``model_parallel_size``, ``inner_mlp_size``),
        read with ``.get()`` so that missing keys silently become ``None``.

    Notes
    -----
    * The block contains no dropout (``MHA`` defaults to ``dropout=0.0``).
    * ``causal=True`` is hard-coded for the inner attention -- see the comment at
      that call site for how this interacts with the manuscript's claim that
      causal masking is omitted.
    """
    def __init__(
        self,
        # basic parameters
        hidden_size: int,
        num_attention_heads: int,
        layer_idx: int,
        # normalization parameters
        use_rms_norm: bool = True,
        
        # projection-group parameters
        proj_groups: int = 1,
        
        # dtype parameters
        attn_block_dtype: torch.dtype = torch.bfloat16,
        mlp_dtype: torch.dtype = torch.bfloat16,
        
        # MHA parameters
        qkv_proj_bias: bool = True,
        mha_out_proj_bias: bool = True,
        use_flash_attn: bool = True,
        rotary_emb_base: float = 1000000,
        
        # rotary positional encoding parameters
        use_interpolated_rotary_pos_emb: bool = False,
        rotary_emb_scaling_factor: float = 1.0,
        
        # GQA parameters
        smeared_gqa: bool = False,
        
        # MLP parameters (passed through to ParallelGatedMLP)
        **kwargs
    ) -> None:
        super().__init__()
        
        # store the basic parameters
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        # head_dim = D / H.  The rotary width passed to MHA below is the full head
        # dimension, so RoPE rotates every channel of Q and K and there is no
        # un-rotated "pass-through" tail.
        self.proj_groups = proj_groups
        
        # initialize the normalization layer
        self.pre_norm = RMSNorm(hidden_size)
        self.post_norm = RMSNorm(hidden_size)
        # Two separate norms with their own gains: pre_norm feeds attention and
        # post_norm feeds the MLP, so the two sub-layers see independently scaled
        # inputs even though they share the same residual stream.
        
        # initialize the counter
        self.counter = 0
        
        # initialize MHA
        self.inner_mha_cls = MHA(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            num_heads_kv=num_attention_heads // proj_groups,
            # num_heads_kv == num_heads when proj_groups == 1 (eScreen), which
            # makes MHA take its "fused QKV" path; proj_groups > 1 would instead
            # allocate fewer K/V heads (grouped-query attention).
            rotary_emb_dim=hidden_size // num_attention_heads,
            qkv_proj_bias=qkv_proj_bias,
            rotary_emb_base=rotary_emb_base,
            causal=True,
            # NOTE -- divergence from Supplementary Methods §2.1: causal=True is
            # hard-coded here.  Because eScreen runs MHA with use_flash_attn=False,
            # this value reaches the PyTorch path
            # F.scaled_dot_product_attention(..., is_causal=True), i.e. the
            # attention scores of this block ARE lower-triangular: position i can
            # only see positions j <= i.  §2.1 states that explicit causal masking
            # was omitted so that both upstream and downstream context are
            # visible; that holds for the three Hyena blocks (their engine is
            # built with bidirectional=True) but not for this block as coded.
            out_proj_bias=mha_out_proj_bias,
            use_flash_attn=use_flash_attn,
        ).to(dtype=attn_block_dtype)
            # Cast the attention sub-module (QKV/out projections and the rotary
            # frequency buffer) to the requested dtype at construction time.

        # if interpolated rotary positional encoding is needed
        if use_interpolated_rotary_pos_emb:
            swap_mha_rope(
                mha=self.inner_mha_cls,
                kwargs_new_rope={"scaling_factor": rotary_emb_scaling_factor},
            )
            # NOTE: ``swap_mha_rope`` is not imported in this module (the imports
            # are RMSNorm / ParallelGatedMLP / MHA only), so enabling
            # use_interpolated_rotary_pos_emb would raise NameError.  eScreen leaves
            # the flag at its default False, so this branch is never taken.

        # if smeared GQA is needed
        if smeared_gqa:
            self.inner_mha_cls.num_heads_kv = self.inner_mha_cls.num_heads
            # NOTE: overwriting num_heads_kv *after* MHA.__init__ cannot resize the
            # already allocated Wqkv weight, whose shape is
            # head_dim * (num_heads + 2 * num_heads_kv).  The assignment is only
            # self-consistent when proj_groups == 1, which is what eScreen uses, so
            # it is effectively a no-op in the deployed model.
            
        # register the inv_freq buffer
        self.inner_mha_cls.rotary_emb.register_buffer("inv_freq", self.inner_mha_cls.rotary_emb.inv_freq)
        # Re-register the inverse-frequency buffer.  RotaryEmbedding registers it
        # with persistent=False, so this call (default persistent=True) is what
        # actually puts "rotary_emb.inv_freq" into this block's state_dict --
        # i.e. it exists for checkpoint-key compatibility with the released
        # weights, not for computation (the values are identical).

        # initialize the MLP
        self.mlp = ParallelGatedMLP(
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            inner_size_multiple_of = kwargs.get("inner_size_multiple_of"),
            mlp_activation = kwargs.get("mlp_activation"),
            evo2_style_activations = kwargs.get("evo2_style_activations"),
            model_parallel_size = kwargs.get("model_parallel_size"),
            inner_mlp_size = kwargs.get("inner_mlp_size")
        ).to(dtype=mlp_dtype)
        # The five MLP hyper-parameters are pulled out of **kwargs with .get(),
        # so an absent key becomes None.  None is *not* the same as "use the
        # default": ParallelGatedMLP does `multiple_of * model_parallel_size`, so a
        # None inner_size_multiple_of would fail; escreen/escreen.py always passes
        # all five explicitly.

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        """Run one attention-block step on the residual stream.

        Parameters
        ----------
        u : torch.Tensor
            Hidden states ``[B, L, D]`` (bfloat16 in eScreen, ``L = 512``).
        inference_params : optional
            Megatron-style decoding cache, forwarded to ``MHA``.  eScreen always
            passes ``None``, so only the full-sequence path is exercised.
        padding_mask : torch.Tensor, optional
            Shape ``[B, L]``; multiplied in (not added) as ``[..., None]`` so
            padded positions are zeroed.  eScreen passes ``None`` because every
            input window has the same fixed length.
        *args, **kwargs
            Ignored; accepted so the block can be called with the same signature
            as ``ParallelGatedConvBlock``.

        Returns
        -------
        torch.Tensor
            ``[B, L, D]``, *including* the two residual additions made in this
            method.

        Notes
        -----
        The caller in ``escreen/escreen.py`` does ``x = x + layer(x)`` for every
        block, so each block's output is added to its input twice in total (once
        inside the block, once in the stack loop).  The same is true of the Hyena
        blocks; see the note in ``ParallelGatedConvBlock.forward``.
        """
        if type(padding_mask) == torch.Tensor:
            u = u * padding_mask[..., None]
            # Zero padded positions *before* the norm + attention so that garbage
            # window padding cannot enter the keys/values of valid positions.

        u = (
            self.inner_mha_cls(
                self.pre_norm(u),
                inference_params=inference_params,
            )
            + u
        )

        if type(padding_mask) == torch.Tensor:
            u = u * padding_mask[..., None]

        u = self.mlp(self.post_norm(u)) + u
        # Both sub-layers are residual: attention output was added to u above,
        # and the gated MLP output is added here.
        return u#, None