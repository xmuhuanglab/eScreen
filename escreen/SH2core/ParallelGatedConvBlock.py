"""Hyena residual block: short depth-wise convolution + implicit long convolution.

Provides :class:`ParallelGatedConvBlock`, the block type used by three of the four
striped blocks in the eScreen backbone -- HyenaSE (``layer_idx = 0``),
HyenaMR (``layer_idx = 1``) and HyenaLI (``layer_idx = 2``) in
``escreen/escreen.py`` ``get_block``.  The three share this class and differ only
in the arguments they pass to :class:`~.HyenaCascade.HyenaCascade`.

Forward computation (pre-norm, with ``D = hidden_size``)::

    z     = Linear3D(pre_norm(u))          # [B, L, 3D] -> split into x2, x1, v
    z     = HyenaCascade(z)                # short conv (L=3) + long conv + gate
    z_in  = out_filter_dense(z) + u        # residual
    y     = MLP(post_norm(z_in)) + z_in    # residual gated MLP

The long-range part of the operator is what gives the block a global receptive
field in a single layer, which is exactly the property §2.1 attributes to
StripedHyena2 ("establish global sequence context within a single module").

eScreen configuration (see ``get_block``): ``hidden_size = d_model`` (128/256),
``state_size = 16``, ``short_filter_length = 3`` (§2.1), ``short_filter_bias =
False``, ``qkv_proj_bias = False``, ``hyena_out_proj_bias = True``,
``hyena_filter_groups = 128``, ``fir_inner_filter_length`` = 7 (HyenaSE) /
128 (HyenaMR) / not passed (HyenaLI), ``column_split_hyena = False``,
``interleave = True``, ``bidirectional = True``, ``use_flashfft = False``,
``use_flash_depthwise = False``, ``compile`` not passed (False).

Adapted from the StripedHyena2 reference implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .HyenaCascade import HyenaCascade
from .RMSNorm import RMSNorm
from .ParallelGatedMLP import ParallelGatedMLP


class ParallelGatedConvBlock(nn.Module):
    """Residual block combining a short depth-wise filter with a Hyena long filter.

    Parameters
    ----------
    hidden_size : int
        Model width ``D``.
    layer_idx : int
        Index of the block in the stack (0..2 for the three Hyena blocks).  Used
        to index the FIR/IIR state dictionaries during incremental decoding.
    qkv_proj_bias : bool
        Bias for the input ``Linear(D, 3D)``.  eScreen: False.
    hyena_out_proj_bias : bool
        Bias for the output ``Linear(D, D)``.  eScreen: True -- this is the only
        bias inside the Hyena branch, which is what lets the filter output be
        re-centred before it is added to the residual stream.
    hyena_filter_groups : int or None, optional
        Number of long-filter "systems".  Defaults to ``hidden_size`` (one
        channel per group); eScreen passes 128, i.e. one filter per channel at
        ``D = 128``.  Must divide ``hidden_size``.
    fir_inner_filter_length : int or None, optional
        Length of the *explicit* long filter.  ``None`` selects the implicit
        (poles/residues) parameterisation instead -- this is the single switch
        that distinguishes HyenaLI from HyenaSE/HyenaMR.
    low_mem_mode : bool, optional
        Stored as an attribute but never read in this class; the flag is only
        consulted inside ``HyenaInferenceEngine.parallel_iir`` when
        ``inference_params`` is supplied (never in eScreen).
    hyena_block_dtype : torch.dtype, optional
        Dtype of the norms, filter and both linear projections (bfloat16).
    mlp_dtype : torch.dtype, optional
        Dtype of the gated MLP sub-module (bfloat16).
    use_fp8_input_projections : bool, optional
        Enables the sequence-length padding in :meth:`pad_to_multiple` required
        by the FP8 input-projection kernel.  False in eScreen, where the padding
        helper is a pure identity.
    compile : bool, optional
        If True, ``torch.compile`` the two norm/projection helpers.  Not passed
        by eScreen, hence False.
    *args, **kwargs
        Every remaining Hyena/MLP hyper-parameter is read from ``**kwargs`` with
        ``.get()`` (``state_size``, ``num_filters``, ``num_attention_heads``,
        ``short_filter_length``, ``use_flashfft``, ``interleave``,
        ``bidirectional``, the five MLP keys, ...).  Missing keys become ``None``,
        which is *not* equivalent to "keep the default" for parameters that are
        consumed arithmetically -- ``escreen/escreen.py`` therefore passes them
        all explicitly.

    Notes
    -----
    With ``column_split_hyena = False`` (eScreen) the ``3 * hidden_size`` channel
    axis is simply ``torch.split`` into three contiguous ``[B, L, D]`` slabs,
    which is the layout the released checkpoints were trained with.
    """
    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        qkv_proj_bias: bool,
        hyena_out_proj_bias: bool,
        hyena_filter_groups: Optional[int] = None,
        fir_inner_filter_length: Optional[int] = None,
        low_mem_mode: bool = False,
        hyena_block_dtype: torch.dtype = torch.bfloat16,
        mlp_dtype: torch.dtype = torch.bfloat16,
        use_fp8_input_projections: bool = False,
        compile: bool = False,
        *args,   # positional arguments not listed above
        **kwargs # keyword arguments not listed above
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.low_mem_mode = low_mem_mode
        self.fir_inner_filter_length = fir_inner_filter_length
        self.hyena_filter_groups = hyena_filter_groups if hyena_filter_groups is not None else hidden_size
        # Default (None) means "one filter system per channel"; eScreen passes 128
        # explicitly.  HyenaCascade asserts hidden_size % num_filters == 0, and
        # channels_per_group = hidden_size // hyena_filter_groups must be >= 1.
        
        self.pre_norm, self.post_norm = (
            RMSNorm(hidden_size).to(dtype=hyena_block_dtype),
            RMSNorm(hidden_size).to(dtype=hyena_block_dtype),
        )
        
        self.filter = HyenaCascade(
            state_size =kwargs.get("state_size"),
            hidden_size=hidden_size,
            num_filters=kwargs.get("num_filters"),
            num_attention_heads=kwargs.get("num_attention_heads"),
            short_filter_length=kwargs.get("short_filter_length"),
            short_filter_bias  =kwargs.get("short_filter_bias"),
            hyena_filter_groups=self.hyena_filter_groups,
            fir_inner_filter_length=fir_inner_filter_length,
            layer_idx=layer_idx,
            use_flashfft        = kwargs.get("use_flashfft"),
            inference_mode      = kwargs.get("inference_mode"),
            column_split_hyena  = kwargs.get("column_split_hyena"),
            hyena_flip_x1x2     = kwargs.get("hyena_flip_x1x2"),
            use_flash_depthwise = kwargs.get("use_flash_depthwise"),
            depthwise_dtype     = kwargs.get("depthwise_dtype"),
            long_fir_threshold  = kwargs.get("long_fir_threshold"),
            interleave          = kwargs.get("interleave"),
            prefill_style       = kwargs.get("prefill_style"),
            bidirectional       = kwargs.get("bidirectional")
        ).to(dtype=hyena_block_dtype)

        self.projections = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_proj_bias).to(dtype=hyena_block_dtype)
        # Single fused projection producing (x2, x1, v) in one matmul; the channel
        # order (first third -> x2, second -> x1, third -> v) is fixed by
        # HyenaCascade / HyenaInferenceEngine.parallel_iir.  bias=False in eScreen:
        # the block's residual stream already carries the information a constant
        # offset would add, and the following norm would absorb most of it.
        self.out_filter_dense = nn.Linear(hidden_size, hidden_size, bias=hyena_out_proj_bias).to(dtype=hyena_block_dtype)
        # Projects the filter output back into the residual width; this is the only
        # biased projection in the block (eScreen passes hyena_out_proj_bias=True).
        self.mlp = ParallelGatedMLP(
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            inner_size_multiple_of = kwargs.get("inner_size_multiple_of"),
            mlp_activation = kwargs.get("mlp_activation"),
            evo2_style_activations = kwargs.get("evo2_style_activations"),
            model_parallel_size = kwargs.get("model_parallel_size"),
            inner_mlp_size = kwargs.get("inner_mlp_size")
        ).to(dtype=mlp_dtype)

        if compile:
            # torch.compile is opt-in and is not enabled by eScreen: the model is
            # trained/inferred in bfloat16 where these two tiny helper functions are
            # already dominated by the kernels they call.
            self.proj_norm_fn = torch.compile(self.proj_norm, fullgraph=True, dynamic=False, mode="reduce-overhead")
            self.res_mlp_norm_fn = torch.compile(
                self.res_mlp_norm, fullgraph=True, dynamic=False, mode="reduce-overhead"
            )
        else:
            self.proj_norm_fn = self.proj_norm
            self.res_mlp_norm_fn = self.res_mlp_norm
            
        self.use_fp8_input_projections = use_fp8_input_projections
        # FP8 support is off in eScreen, so pad_to_multiple below returns its input
        # unchanged and no sequence padding ever happens.

    def pad_to_multiple(self, x, multiple=16):
        """Right-pad the sequence axis to a multiple of ``multiple``.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, L, D]``.
        multiple : int, optional
            Required alignment of ``L`` (default 16).

        Returns
        -------
        torch.Tensor
            ``x`` unchanged when FP8 input projections are disabled (eScreen), or
            ``[B, L', D]`` with ``L' >= L``, ``L' % multiple == 0`` otherwise.

        Notes
        -----
        The FP8 GEMM used by the upstream model requires the flattened token count
        to be a multiple of 16; instead of padding at the batch level the helper
        pads the sequence axis and :meth:`proj_norm` trims the result back.
        """
        """Pad input tensor to multiple of 16 only when FP8 is enabled"""
        # The moment this flag is False (the eScreen case) this method is an
        # identity, so L stays 512 and the trim in proj_norm is a no-op.
        if not self.use_fp8_input_projections:
            return x

        batch_size, seq_len, hidden_dim = x.size()
        pad_len = (multiple - (seq_len % multiple)) % multiple
        if pad_len == 0:
            return x
        return F.pad(x, (0, 0, 0, pad_len))
        # F.pad's tuple is (dim=-1 left, right, dim=-2 left, right), i.e. feature
        # axis untouched and the *sequence* axis padded on the right only -- so the
        # first L positions still correspond to the original tokens.

    def proj_norm(self, x):
        """Normalise, optionally pad, and apply the fused 3-way input projection.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, L, D]`` residual-stream input.

        Returns
        -------
        torch.Tensor
            ``[B, L, 3 * D]`` projections (padded back to the original ``L``).

        Notes
        -----
        The hidden dimension is *not* changed by ``pad_to_multiple`` (only the
        sequence axis is padded), so a non-FP8 run behaves as
        ``Linear(D, 3D)(RMSNorm(x))``.
        """

        normalized = self.pre_norm(x)
        normalized = self.pad_to_multiple(normalized)
        with torch.cuda.device(x.device):
            # Scopes the projection to the tensor's CUDA device.  Redundant in eScreen,
            # where the whole backbone lives on one device; it exists because the
            # optional FP8/multi-device projection kernels dispatch on the *current*
            # CUDA device rather than on the input's.
            projected = self.projections(normalized)

        if isinstance(projected, tuple):
            projected = projected[0]

        original_seq_len = x.size(1)
        # Only reachable on the FP8 path: drop the positions added by
        # pad_to_multiple so the caller sees exactly L outputs.
        if projected.size(1) > original_seq_len:
            projected = projected[:, :original_seq_len, :]

        return projected

    def res_mlp_norm(self, x):
        """Gated-MLP sub-layer with its own pre-norm: ``post_norm(x) -> MLP + x``.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, L, D]`` (the Hyena branch output, already carrying the outer
            residual).

        Returns
        -------
        torch.Tensor
            ``[B, L, D]``.
        """
        return self.mlp(self.post_norm(x)) + x

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        """Run one Hyena block step on the residual stream.

        Parameters
        ----------
        u : torch.Tensor
            Hidden states ``[B, L, D]`` (bfloat16, ``L = 512`` in eScreen).
        inference_params : optional
            Decoding cache.  ``HyenaCascade`` switches to its recurrent
            ``sequential_forward`` when this is not None; eScreen always passes
            ``None`` and takes the parallel (whole-sequence) path.
        padding_mask : torch.Tensor, optional
            ``[B, L]`` multiplicative mask; applied before the filter, after the
            residual and inside ``HyenaInferenceEngine.parallel_fir``.  Always
            ``None`` in eScreen (fixed-length windows).
        *args, **kwargs
            Ignored; present so all four block types share one call signature.

        Returns
        -------
        torch.Tensor
            ``[B, L, D]``, including both residual additions made here.

        Notes
        -----
        ``escreen/escreen.py`` adds the block input again in its layer loop
        (``x = x + layer(x)``), so the deployed model applies a double residual
        around every block -- including this one, which already returns
        ``MLP(...) + z_in``.  The block output is *not* RMS-normalised before
        leaving, so the residual stream scale grows across the four blocks; the
        pre-norms at the start of each sub-layer absorb that.
        """
        z = self.proj_norm_fn(u)

        if type(padding_mask) == torch.Tensor:
            z = z * padding_mask[..., None]

        z, inference_params = self.filter(z, inference_params=inference_params, padding_mask=padding_mask)

        z_in = self.out_filter_dense(z) + u
        # Residual around the whole Hyena branch: the inner projections have no
        # bias in eScreen (qkv_proj_bias=False), so adding u is what keeps the
        # branch zero-centred when it is initialised.

        if type(padding_mask) == torch.Tensor:
            z_in = z_in * padding_mask[..., None]

        y = self.res_mlp_norm_fn(z_in)

        return y#, inference_params