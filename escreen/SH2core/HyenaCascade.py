"""The Hyena operator: short depth-wise filter + long convolution + element-wise gate.

Provides the module-level helper :func:`interleave` and the class
:class:`HyenaCascade`, which is the "filter" half of every
``ParallelGatedConvBlock`` in the eScreen backbone (blocks 0-2: HyenaSE,
HyenaMR, HyenaLI, see ``escreen/escreen.py`` ``get_block``).

What the operator computes
--------------------------
Given the ``[B, L, 3D]`` projection output split into ``x2``, ``x1`` and ``v``::

    # 1. short depth-wise filter (length 3 in eScreen, §2.1)
    z_pre = depthwise_conv1d(u, short_filter_weight)      # 3D channels, no mixing
    # 2. gated long filter, either explicit (FIR) or implicit (poles/residues)
    y = (long_filter(x1 * v) + D * (x1 * v)) * x2
    return Linear(y)                                      # done by the block

Two parameterisations of the long filter are supported and exactly one is
selected by ``fir_inner_filter_length``:

* **explicit FIR** -- a learnable kernel ``h`` of length
  ``fir_inner_filter_length``, applied by FFT when the length is >= 128
  (HyenaMR, length 128) and by a plain depth-wise ``conv1d`` when it is short
  (HyenaSE, length 7);
* **implicit / modal** -- ``fir_inner_filter_length = None`` (HyenaLI) stores a
  bank of ``state_size`` log-poles and residues instead, and the long filter is
  materialised as the impulse response
  ``h[t] = sum_k residues[k] * exp(log_poles[k] * t)``.
  ``state_size = 16`` in eScreen, as stated in Supplementary Methods §2.1.

Bidirectionality
----------------
``bidirectional=True`` is passed down to ``HyenaInferenceEngine.parallel_fir``,
where it only affects the **FFT branch** (``fir_length >= 128``): there the
kernel is split along the time axis and its second half is applied to the
time-reversed input (implemented as a conjugation in the frequency domain), so
one filter simultaneously integrates upstream and downstream context.  The short
``conv1d`` branch (used by HyenaSE) ignores the flag and left-pads by
``fir_length - 1``, i.e. that inner filter is effectively causal -- see the note
in :meth:`parallel_forward`.

Adapted from the StripedHyena2 reference implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .HyenaInferenceEngine import HyenaInferenceEngine

try:
    from flashfftconv import FlashDepthWiseConv1d
except:
    assert 1==0,'Could not import flashfftconv'


    
    
def interleave(z_pre):
    """Reorder the ``x1``/``x2``/``v`` channel blocks, preserving their layout.

    Parameters
    ----------
    z_pre : torch.Tensor
        Either ``[B, L, 3D]`` ("non-cached", sequence axis in the middle) or
        ``[B, D, L]`` (cached/step layout, sequence axis last).

    Returns
    -------
    torch.Tensor
        Same shape, with the three ``D``-wide channel blocks re-laid out as
        ``(x1 block, x2 block, v block)`` -- i.e. the interleaving of every third
        slice is a no-op on the values, it only documents the intended block order.

    Notes
    -----
    ``0::3 / 1::3 / 2::3`` take every third position, so ``x1``, ``x2`` and ``v``
    are gathered from strided positions and then concatenated: this is the layout
    produced by the upstream (column-interleaved) checkpoint, not by the
    contiguous ``torch.split`` used elsewhere in this file.  The function's name
    is also used as the parameter name of ``HyenaCascade.__init__``; inside that
    constructor the name refers to the bool, and it is only the module-level
    function that is called from ``forward``.
    """
    if len(z_pre.shape) == 3:  # non-cached
        x1 = z_pre[:, 0::3, :]
        x2 = z_pre[:, 1::3, :]
        v = z_pre[:, 2::3, :]
        z_pre = torch.cat([x1, x2, v], dim=1)
        return z_pre
    else:
        x1 = z_pre[..., 0::3]
        x2 = z_pre[..., 1::3]
        v = z_pre[..., 2::3]
        z_pre = torch.concat([x1, x2, v], dim=-1)
        return z_pre
    
    
class HyenaCascade(nn.Module):
    """Short depth-wise filter followed by a gated long (Hyena) convolution.

    Input/output shape: ``[B, L, 3D] -> [B, L, D]`` where ``D = hidden_size`` and
    the ``3D`` input is the fused ``(x2, x1, v)`` projection produced by
    ``ParallelGatedConvBlock.projections``.  Internally the tensor is transposed
    to ``[B, 3D, L]`` for the convolutions.

    Parameters
    ----------
    state_size : int
        Number of poles/residues of the implicit long filter; 16 in eScreen
        (§2.1).  Only used when ``fir_inner_filter_length`` is None.
    hidden_size : int
        Model width ``D``.
    num_filters : int
        Number of filter systems; asserted to divide ``hidden_size``.  eScreen
        passes ``d_model``, i.e. one system per channel.
    num_attention_heads : int
        Not used for attention here: it only defines the "column split" layout
        (``hidden_size_per_attention_head``) that some checkpoints use to store
        the ``3D`` projections, and the head-wise reshaping in
        ``HyenaInferenceEngine.parallel_iir`` when ``column_split_hyena`` is True.
    short_filter_length : int
        Length of the depth-wise filter; 3 in eScreen (§2.1).
    short_filter_bias : bool, optional
        Add a learnable per-channel bias after the short filter.  eScreen: False,
        so ``short_filter_bias`` is ``None`` and the attribute exists but is unused.
    hyena_filter_groups : int, optional
        Number of long-filter systems that are shared across channels; eScreen
        passes 128, so at ``D = 128`` every channel has its own filter.
    fir_inner_filter_length : int or None, optional
        ``None`` -> implicit modal parameterisation; otherwise an explicit FIR of
        that length.  This single argument separates HyenaSE (7), HyenaMR (128)
        and HyenaLI (None).
    layer_idx : int, optional
        Block index, used as the key of the FIR/IIR state dicts during decoding.
    use_flashfft : bool, optional
        Use the fused flashfftconv kernel for the long convolution.  eScreen:
        False, so the plain FFT path in ``HyenaInferenceEngine.parallel_iir`` runs.
    inference_mode : bool, optional
        Stored but never read in this class.
    column_split_hyena : bool, optional
        If True, reinterpret the ``3D`` axis as ``H`` groups of
        ``3 * D/H`` (checkpoint layout) instead of three contiguous ``D`` slabs.
        eScreen: False.  NOTE: the ``True`` path of ``sequential_forward`` calls a
        ``column_split`` helper that is not imported in this module, so it would
        raise ``NameError`` -- it is unreachable in eScreen.
    hyena_flip_x1x2 : bool, optional
        Swap the roles of ``x1`` and ``x2`` (checkpoint convention).  eScreen: False.
    use_flash_depthwise : bool, optional
        Use ``FlashDepthWiseConv1d`` for the short filter.  eScreen: False, so
        ``F.conv1d`` is used and ``fir_inner_fn`` is defined.
    depthwise_dtype : torch.dtype, optional
        dtype passed to the flash depth-wise kernel (unused when the flag is False).
    long_fir_threshold : int or None, optional
        If set, very long explicit filters are convolved in the time domain
        instead of by FFT.  Unused in eScreen (None); asserted incompatible with
        ``use_flashfft``.
    interleave : bool, optional
        Apply :func:`interleave` to the projected tensor.  eScreen: True.
    prefill_style : str, optional
        Passed to the engine for KV-cache prefill; not used by the training /
        full-sequence path that eScreen runs.
    bidirectional : bool, optional
        Passed to the engine; only the FFT branch of the long filter honours it.

    Notes
    -----
    The class is *not* a full residual block: it has no norm and no residual
    connection.  Those live in ``ParallelGatedConvBlock``.
    """
    def __init__(
        self,
        # Config parameters
        state_size: int,
        hidden_size: int,
        num_filters: int,
        num_attention_heads: int,
        short_filter_length: int,
        short_filter_bias: bool = False,
        # Optional parameters with defaults
        hyena_filter_groups: int = 1,
        fir_inner_filter_length: int = None,
        layer_idx: int = 0,
        use_flashfft: bool = False,
        inference_mode: bool = True,
        column_split_hyena: bool = True,
        hyena_flip_x1x2: bool = False,
        use_flash_depthwise: bool = False,
        depthwise_dtype: torch.dtype = torch.bfloat16,
        long_fir_threshold: int = None,
        interleave: bool = False,
        prefill_style: str = "fft",
        bidirectional: bool = True
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hyena_filter_groups = hyena_filter_groups

        self.use_flashfft = use_flashfft
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.inference_mode = inference_mode
        self.counter = 0
        self.column_split_hyena = column_split_hyena
        self.hyena_flip_x1x2 = hyena_flip_x1x2

        assert self.hidden_size % self.num_filters == 0 and self.num_filters <= self.hidden_size
        # One filter system per channel requires D % num_filters == 0; with
        # num_filters == hidden_size (eScreen) every channel gets its own system.
        # channels_per_group is used by the engine to expand shared filters.

        # attention heads are not used except to split post short_filter
        # projections in the same way as the checkpoint
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads
        # head_dim of the "virtual" attention split; only used to reproduce the
        # checkpoint's column layout when column_split_hyena is True.

        self.fir_inner_filter_length = fir_inner_filter_length
        self.short_filter_length = short_filter_length
        self.short_filter_weight = nn.Parameter(torch.randn(3 * hidden_size, 1, short_filter_length))
        # Short depth-wise filter: shape [3D, 1, L_short] so that F.conv1d with
        # groups = 3D mixes no channels (one kernel tap-set per projection
        # channel).  The upstream code uses randn rather than a variance-scaled
        # init; the following RMSNorm renormalises the result, so the absolute
        # scale of this weight is not critical.
        self.short_filter_bias = nn.Parameter(torch.randn(3 * hidden_size)) if short_filter_bias else None
        # Optional bias on the short filter.  eScreen passes short_filter_bias=False
        # (2.1 lists only the depth-wise filter, not a bias), so this stays None;
        # the code below handles None by passing bias=None to F.conv1d.

        self.engine = HyenaInferenceEngine(
            layer_idx=layer_idx,
            hyena_flip_x1x2=hyena_flip_x1x2,
            bidirectional=bidirectional
        )
        self.use_flash_depthwise = use_flash_depthwise
        self.data_dtype = None

        if self.use_flash_depthwise:
            try:
                self.fir_fn = FlashDepthWiseConv1d(
                    channels=3 * self.hidden_size,
                    kernel_size=self.short_filter_length,
                    padding=self.short_filter_length - 1,
                    weights=self.short_filter_weight,
                    bias=self.short_filter_bias,
                    device=None,
                    dtype=depthwise_dtype,
                )
            except ImportError:
                "flashfftconv not installed"
                self.fir_fn = F.conv1d
                # The bare string on the line above is an expression statement (a
                # no-op, not a comment); this assignment is the real fallback taken
                # when flashfftconv is missing.
        else:
            self.fir_fn = F.conv1d
            self.fir_inner_fn = F.conv1d
            # eScreen path: plain depth-wise conv1d for the short filter and for a
            # short explicit inner filter (HyenaSE).  fir_inner_fn is only defined
            # in this branch, which is why the FFT branch of parallel_fir is guarded
            # by fir_length instead.

        self.fftconv_fn = None
        self.long_fir_threshold = long_fir_threshold
        if self.long_fir_threshold is not None:
        # Time-domain convolution for very long explicit filters (memory saver for
        # 100k+ contexts).  Nothing in eScreen sets it, and the assertion below
        # keeps it mutually exclusive with the fused flashfft kernel.
            assert self.use_flashfft is False, "long_fir_threshold not compatible with fused flashfft"

        self.num_systems = self.hyena_filter_groups
        self.channels_per_group = self.hidden_size // self.hyena_filter_groups
        # Number of long-filter systems and how many channels each one serves;
        # both are forwarded to the engine through the 5-tuple `dims` below.
        self.interleave = interleave
        self.prefill_style = prefill_style
        self.bidirectional = bidirectional

        if self.fir_inner_filter_length:
            self.h = nn.Parameter(torch.randn(self.hyena_filter_groups, 1, fir_inner_filter_length))
            # Explicit FIR branch: one learnable kernel per filter group, shape
            # [groups, 1, L_fir].  Randn init again (see the short filter above).

            if fir_inner_filter_length >= 128:
                self.D = nn.Parameter(torch.zeros(self.hidden_size))
                # D is a learnable skip / feed-through gain, only present for the
                # FFT parameterisation (L >= 128).  Zero-init so the block starts as
                # a pure convolution: the skip path contributes nothing until
                # training moves it away from 0.
            else:
                self.D = None
        else:
            log_poles = -1*torch.abs(torch.randn(self.num_systems, self.state_size, 1, dtype=torch.float32)) # initialization happens here
            # Implicit (modal) branch, used by HyenaLI.  Each pole is initialised on
            # the negative real axis as -|N(0,1)| so that exp(log_pole * t) decays
            # with time (a stable, causal IIR system); a positive pole would blow up
            # over the 512 positions.  state_size = 16 systems per channel group.
            self.log_poles = nn.Parameter(log_poles) # log_poles is registered here as a learnable parameter
            self.residues = nn.Parameter(torch.randn(self.num_systems, self.state_size, dtype=torch.float32))
            self.D = nn.Parameter(torch.zeros(self.hidden_size))
            # Feed-through gain for the modal branch, zero-init for the same reason
            # as above: start purely convolutional, let training add the skip.
            self.h = None
        self.t = None

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        """Dispatch to the parallel (whole-sequence) or step-wise forward pass.

        Parameters
        ----------
        u : torch.Tensor
            ``[B, L, 3D]`` projections (bfloat16).
        inference_params : optional
            Decoding cache.  If it already holds FIR state for this layer, the
            single-token :meth:`sequential_forward` path is used; otherwise
            :meth:`parallel_forward`.
        padding_mask : torch.Tensor, optional
            ``[B, L]`` mask forwarded to the engine.  Always None in eScreen.
        *args, **kwargs
            Ignored (accepts the generic block call signature).

        Returns
        -------
        tuple
            ``(y, inference_params)`` with ``y`` of shape ``[B, L, D]``.  Note the
            truthiness test on ``inference_params`` below: a plain dict-like
            object that compares False would silently take the parallel path.
        """
        if inference_params is not None and self.layer_idx in inference_params.fir_state_dict.keys():
            return self.sequential_forward(u, inference_params)
        else:
            return self.parallel_forward(u, inference_params, padding_mask)

    def parallel_forward(self, u, inference_params=None, padding_mask=None):
        """Whole-sequence (training / prefill) forward pass.

        Parameters
        ----------
        u : torch.Tensor
            ``[B, L, 3D]``.
        inference_params : optional
            If given, the FIR cache for this layer is written into
            ``inference_params.fir_state_dict[layer_idx]``.
        padding_mask : torch.Tensor, optional
            ``[B, L]`` multiplicative mask.

        Returns
        -------
        tuple
            ``(y, inference_params)``, ``y`` of shape ``[B, L, D]``.

        Notes
        -----
        * ``L`` is re-read from ``u`` inside the engine, so the ``L`` local here is
          only used for the FFT filter length.
        * The 5-tuple ``dims`` is
          ``(hidden_size, num_attention_heads, head_dim, state_size,
          hyena_filter_groups)`` and is unpacked positionally by the engine.
        * ``gate=False`` for the short filter: the gating multiplication happens
          *after* the long filter, on ``(x1 * v)``, not on the raw projection.
        """
        L = u.shape[1]
        dims = (
            self.hidden_size,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.state_size,
            self.hyena_filter_groups,
        )

        z_pre, fir_state = self.engine.parallel_fir(
            self.fir_fn,
            u,
            self.short_filter_weight,
            self.short_filter_bias,
            L,
            dims=dims,
            gate=False,
            column_split_hyena=self.column_split_hyena,
            fir_length=self.short_filter_length,
            inference_params=inference_params,
            padding_mask=padding_mask,
            dim_last=True,
            bidirectional = self.bidirectional
        )
        # The short filter is applied to all 3D channels at once via F.conv1d with
        # groups = 3D (see the engine), i.e. strictly per-channel.  `bidirectional`
        # is handed to the engine but is only honoured when the inner filter is
        # long enough to take the FFT branch.

        if inference_params:
            inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.interleave:
            z_pre = interleave(z_pre)

        if self.h is None:
        # HyenaLI: no explicit kernel was stored, so the long filter is the impulse
        # response of the 16-system modal bank, materialised here in float32.
            h, _, _, _ = self.compute_filter(L, u.device)
        else:
            h = self.h

        D = self.D

        if self.hyena_filter_groups > 1:
            h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)
        # Each filter groups serves hidden_size // groups consecutive channels
        # (1 channel per group when groups == D, i.e. eScreen's setting).

        if self.fir_inner_filter_length is not None:
        # Two mutually exclusive long-filter parameterisations; exactly one of
        # `fir_inner_filter_length` / `log_poles` exists (see __init__).

            y, fir_inner_state = self.engine.parallel_fir(
                self.fir_inner_fn,
                z_pre,
                h,
                D,
                L,
                dims=dims,
                gate=True,
                gated_bias=self.fir_inner_filter_length >= 128,
                # gated_bias is only enabled for L >= 128, i.e. for the branch whose
                # D parameter exists (None otherwise) -- see the elif in __init__.
                dim_last=False,
                column_split_hyena=self.column_split_hyena,
                fir_length=self.fir_inner_filter_length,
                inference_params=inference_params,
                padding_mask=padding_mask,
                groups=self.hyena_filter_groups,
                bidirectional=self.bidirectional
            )

            y = y.permute(0, 2, 1)
            if inference_params:
                inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            
            y = self.engine.parallel_iir(
                z_pre,
                h,                       # h is also always None when fir_inner_fn is not provided
                D,                       # D is a learnable parameter of size (hidden_size,)
                L,                       # L is the input sequence length
                t=self.t,                # t is always None anyway
                poles=self.log_poles,
                residues=self.residues,
                dims=dims,
                inference_params=inference_params,           # inference_params is also always None
                layer_idx=self.layer_idx,                    # layer_idx is an index and does not take part in computation
                prefill_style=self.prefill_style,            # not used in computation
                use_flashfft=self.use_flashfft,              # not used in computation
                fftconv_fn=self.fftconv_fn,                  # not used in computation
                column_split_hyena=self.column_split_hyena,  # not used in computation
                long_fir_threshold=self.long_fir_threshold,  # not used in computation
                padding_mask=padding_mask,                   # padding_mask is also always None
            )

        return y, inference_params
        # Returns (y, inference_params) even though the caller in
        # ParallelGatedConvBlock unpacks both, so this tuple shape is load-bearing.

    def sequential_forward(self, u, inference_params):
        """Single-token forward pass used during incremental decoding.

        Parameters
        ----------
        u : torch.Tensor
            ``[B, 1, 3D]`` (or ``[B, 3D]``); only the last position is used.
        inference_params : object
            Must carry ``fir_state_dict[layer_idx]`` (and, for the modal branch,
            ``state_dict[layer_idx]``).

        Returns
        -------
        tuple
            ``(y[:, None], inference_params)`` with ``y`` of shape ``[B, D]``
            widened to ``[B, 1, D]`` so it matches the parallel path.

        Notes
        -----
        * ``self.data_dtype`` is latched from the first input and used to cast the
          output back, because the recurrent kernels run in float32.
        * ``column_split_hyena=True`` would call an unimported ``column_split``
          helper here (NameError); eScreen runs with it False, so only the
          ``torch.split`` branch is reachable.
        * The FFT path is never used for a single step, which is why the step code
          only ever calls ``step_fir`` / ``step_iir``.
        """
        if self.data_dtype is None:
            self.data_dtype = u.dtype

        if len(u.shape) > 2:
            u = u[:, -1]

        z_pre, fir_state = self.engine.step_fir(
            u,
            inference_params.fir_state_dict[self.layer_idx],
            weight=self.short_filter_weight,
            bias=self.short_filter_bias,
        )
        inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.interleave:
            z_pre = interleave(z_pre)

        x2, x1, v = (
            column_split(z_pre, self.num_attention_heads, self.hidden_size_per_attention_head)
            if self.column_split_hyena
            else z_pre.split([self.hidden_size, self.hidden_size, self.hidden_size], dim=1)
        )

        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        if self.fir_inner_filter_length is not None:
            if self.hyena_filter_groups > 1:
                h = self.h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)
            else:
                h = self.h

            y, fir_inner_state = self.engine.step_fir(
                x1 * v,
                inference_params.fir_inner_state_dict[self.layer_idx],
                weight=h,
                bias=self.D,
                flip_filter=self.fir_inner_filter_length >= 128,
                gated_bias=self.fir_inner_filter_length >= 128,
            )
            y = y * x2
            inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            y, iir_state = self.engine.step_iir(
                x2,
                x1,
                v,
                self.D,
                self.residues,
                self.log_poles,
                inference_params.state_dict[self.layer_idx],
                iir_groups=1,
            )
            inference_params.state_dict[self.layer_idx] = iir_state

        y = y.to(dtype=self.data_dtype)
        return y[:, None], inference_params

    def update_time(self, L, device, max_length=1024):
        """Cache the position vector ``t = [0 .. L-1]`` used to build the long filter.

        Parameters
        ----------
        L : int
            Required length.
        device : torch.device
            Device for the cached tensor.
        max_length : int, optional
            Accepted but unused; the cache is resized to the requested ``L`` only.

        Returns
        -------
        None
            The tensor is stored on ``self.t`` (shape ``[1, 1, L]``).

        Notes
        -----
        The cache grows on demand and is trimmed (not reallocated) when a shorter
        sequence is requested; ``self.t`` is a plain attribute, not a buffer, so it
        is not part of the state_dict.
        """
        if self.t is None:
            self.t = torch.arange(L, device=device)[None, None]
        elif self.t.shape[-1] < L:
            self.t = torch.arange(L, device=device)[None, None]
        else:
            self.t = self.t[..., :L]

    def compute_filter(self, L, device):
        """Materialise the implicit long filter as an explicit impulse response.

        Parameters
        ----------
        L : int
            Sequence length (filter length).
        device : torch.device
            Device of the position vector.

        Returns
        -------
        tuple
            ``(h, filter_dtype, log_poles, residues)`` where ``h`` has shape
            ``[1, groups, L]`` (float32) and the other three items are only used by
            callers that need the modal parameters (none in eScreen).

        Notes
        -----
        The filter is a sum of ``state_size`` damped exponentials, evaluated in
        float32 regardless of the module dtype: with ``t`` up to 511 and bf16
        (8 bits of mantissa) the products ``log_poles * t`` would lose enough
        precision to visibly bias the filter.
        """
        self.update_time(L, device)
        filter_dtype = torch.float32
        residues, log_poles = (
            self.residues.to(filter_dtype),
            self.log_poles.to(filter_dtype),
        )
        
        h = (residues[..., None] * (log_poles * self.t).exp()).sum(1)[None]  # B, D, L ## this may overflow if t is large
        # sum_k residues[k] * exp(log_poles[k] * t) evaluated in float32.  Because
        # log_poles is initialised negative and stays negative in practice this
        # decays towards 0; the upstream comment warns about overflow, which is the
        # risk if training ever drives a pole positive (exp grows as exp(+t)).
        return h, filter_dtype, log_poles, residues