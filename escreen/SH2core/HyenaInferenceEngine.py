"""Kernels behind the Hyena operator: FFT convolutions and FIR/IIR state handling.

Provides

* :func:`fftconv_standard` -- standalone FFT cross-correlation used as the
  reference long convolution; ``escreen/escreen.py`` imports it (though the
  current forward pass does not call it).
* :func:`fftconv_func` -- the (optionally bidirectional) FFT convolution used by
  the long filter of every ``HyenaCascade``.
* :class:`HyenaInferenceEngine` -- dispatcher that applies the short FIR filter
  (``parallel_fir`` / ``step_fir``), the long IIR filter (``parallel_iir`` /
  ``step_iir``) and builds decoding caches (``prefill_via_*``; most raise
  ``NotImplementedError``).
* :class:`HyenaFilter` -- a vestigial helper class; the file ends in the middle of
  it (see the note there), so it is incomplete and unused.

Role in eScreen
---------------
Each ``HyenaCascade`` (hence each of the three Hyena blocks) owns one engine.  Only
the *parallel* whole-sequence path is ever exercised, because eScreen never passes
``inference_params``; the recurrent/step paths exist for autoregressive decoding.

Numerical conventions
---------------------
* Filters are applied as correlations: the kernel spectrum is conjugated, and the
  result is truncated to the causal part.  A plain conv1d is used with
  ``padding = kernel_len - 1`` and then ``[..., :L]``, which is the same
  left-aligned (causal) window.
* All filter multiplications are promoted to float32 and cast back to the input
  dtype, so the bfloat16 model of §2.1 keeps a numerically stable filter response.

Adapted from the StripedHyena2 reference implementation.
"""

import gc

import torch
import torch.nn.functional as F

from torch.fft import rfftn,irfftn
from functools import partial





def column_split(x, num_heads, head_size):
    """Split a tensor with `num_heads` alongside the head dimension, instead of
    across heads. Fixed to three projections
    """
    # NOTE the ordering: the first third of the channel axis becomes x2, the second
    # becomes x1 and the third becomes v -- i.e. x2 comes *before* x1 here, matching
    # the checkpoint layout, not the (x1, x2, v) reading order used in the comments.
    # FIXME: merge cases
    if len(x.shape) == 2:
        x_reshaped = x.reshape(
            x.shape[0],
            num_heads,
            3 * head_size,
        )

        x2, x1, v = (
            x_reshaped[..., :head_size],
            x_reshaped[..., head_size : 2 * head_size],
            x_reshaped[..., 2 * head_size :],
        )
        x2, x1, v = (
            x2.reshape(x2.shape[0], -1),
            x1.reshape(x1.shape[0], -1),
            v.reshape(v.shape[0], -1),
        )
        return x2, x1, v
    else:
        x = x.reshape(
            x.shape[0],
            num_heads,
            3 * head_size,
            x.shape[2],
        )
        x2, x1, v = (
            x[:, :, :head_size],
            x[
                :,
                :,
                head_size : 2 * head_size,
            ],
            x[:, :, 2 * head_size :],
        )
        x2, x1, v = (
            x2.reshape(x2.shape[0], -1, x2.shape[-1]),
            x1.reshape(x1.shape[0], -1, x1.shape[-1]),
            v.reshape(v.shape[0], -1, v.shape[-1]),
        )
        return x2, x1, v
    
    
    
IIR_PREFILL_MODES = [
    "recurrence",
    "modal-fft",
    "hybrid-modal-recurrence",
    "modal-scan",
    "canonical-fft",
    "iir-fir-caching",
]
# Accepted values of the KV-cache prefill strategy.  Only "modal-fft" and
# "recurrence" have implementations; the rest raise NotImplementedError inside
# HyenaInferenceEngine.  HyenaCascade never passes one, so the default
# "modal-fft" applies -- and in eScreen even that is never reached, because no
# inference_params object is ever supplied.


def adjust_filter_shape_for_broadcast(u, h):
    """Reshape a filter spectrum so that it broadcasts against the signal.

    Parameters
    ----------
    u : torch.Tensor
        Signal used only for its rank: ``[B, D, L]`` or ``[B, D1, D2, L]``.
    h : torch.Tensor
        Filter spectrum with shape ``[1, D, L]`` or ``[D, 1, L]`` (or unbatched
        ``[D, L]``).

    Returns
    -------
    torch.Tensor
        ``h`` with a leading batch axis when ``u`` is batched (and an extra
        broadcast axis for the 4-D case).

    Notes
    -----
    ``h.squeeze()`` collapses both supported layouts to ``[D, L]``; that is only
    safe because exactly one of the two middle axes is always 1 in the callers.
    """
    h = h.squeeze()  # Standardize to [D, L] from [1, D, L] and [D, 1, L]

    # Case: u: [B, D, L], k_f: [D, L]
    if len(u.shape) > len(h.shape):
        h = h.unsqueeze(0)

    # Case: u: [B, D1, D2, L], k_f: [B, D, L]
    if len(u.shape) > 3:
        h = h.unsqueeze(1)
    return h




def complex_matmul(a, b, device='cpu'):
    """Multiply two complex tensors with batched (batched-over-``...``) matmul.

    Parameters
    ----------
    a : torch.Tensor
        complex64 of shape ``[A, ..., L]``.
    b : torch.Tensor
        complex64 of shape ``[C, ..., L]``.
    device : str or torch.device, optional
        Device for the output buffer (must match the inputs).

    Returns
    -------
    torch.Tensor
        complex64 of shape ``[A, C, ..., L]``.

    Notes
    -----
    The einsum pattern ``"ab..., cb... -> ac..."`` treats the first two axes as the
    matrix axes -- the opposite of the usual convention -- which is why both
    callers pass spectra whose leading axis is the (size-1) kernel axis.  Complex
    multiplication is written out as four real products because a complex einsum
    is not a single fused kernel.  The result is assembled through a freshly
    allocated real/imag pair, so the function is not autograd-friendly for complex
    leaves but is safe for the convolution use here.
    """
    # The original one-line summary is retained verbatim on the next line; it is now
    # a no-op string expression, and the NumPy-style docstring above is what Python
    # exposes as this function's __doc__.
    """Multiplies two complex-valued tensors."""
    scalar_matmul = partial(torch.einsum, "ab..., cb... -> ac...")

    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=device)
    c.real, c.imag = real, imag

    return c

def fftconv_standard(signal, kernel, bias = None, padding = 0, device='cpu'):
    """
    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """
    # Conjugating the kernel spectrum turns the circular *convolution* produced by
    # the FFT product into a circular *correlation*; the truncation below then keeps
    # the L_signal - L_kernel + 1 fully-overlapping (causal) positions.  The
    # `padding` argument only widens the analysed window, it does not shift the
    # alignment of the result.
    # 1. Pad the input signal & kernel tensors
    signal = F.pad(signal, [padding, padding])
    kernel_padding = [0, signal.size(-1) - kernel.size(-1)]
    padded_kernel = F.pad(kernel, kernel_padding)
    
    # 2. Perform fourier convolution
    signal_fr = rfftn(signal, dim=-1)
    kernel_fr = rfftn(padded_kernel, dim=-1)

    # 3. Multiply the transformed matrices
    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, device=device)

    # 4. Compute inverse FFT, and remove extra padded values
    output = irfftn(output_fr, dim=-1)
    output = output[:, :, :signal.size(-1) - kernel.size(-1) + 1]

    # 5. Optionally, add a bias term before returning.
    if bias is not None:
        output += bias.view(1, -1, 1)

    return output


def fftconv_func(
    u,
    k,
    D,
    dropout_mask,
    gelu=True,
    k_rev=None,
    bidirectional=True,
    layer_idx=None,
    **kwargs,
):
    """FFT convolution of ``u`` with a learnable filter, optionally two-sided.

    Parameters
    ----------
    u : torch.Tensor
        ``[B, D, L]`` input signal (float32 by the time it arrives here).
    k : torch.Tensor
        ``[1, D, L]`` (or ``[D, L]``) learnable filter.
    D : torch.Tensor
        ``[D]`` feed-through gain applied as ``u * D`` and added to the result.
    dropout_mask : None
        Unused; kept for upstream signature compatibility.
    gelu : bool, optional
        Unused flag -- no activation is applied inside this function.
    k_rev : torch.Tensor, optional
        Time-reversed filter; only used when ``bidirectional`` is False, where its
        conjugated spectrum is added to the forward kernel.
    bidirectional : bool, optional
        If True, the filter is split along its time axis: the first half is applied
        forward and the second half is applied to the time-reversed signal
        (``conj(u_f) * conj(k2_f)``).  This is what lets one long filter integrate
        both upstream and downstream context -- the mechanism behind the
        "no explicit causal mask" design of §2.1.
    layer_idx : int, optional
        Unused here (debug hook).
    **kwargs
        Absorbed and ignored (``groups`` is passed by the caller but not used).

    Returns
    -------
    torch.Tensor
        ``[B, D, L]``, cast back to ``u.dtype``.

    Notes
    -----
    * ``fft_size = 2 * L``: a length-L circular convolution would wrap the filter
      tail onto the head of the signal, so both operands are zero-padded to 2L and
      only the first L output samples are kept (the linear, causal part).
    * The ``/ fft_size`` on the kernel spectrum and ``norm="forward"`` on the inverse
      transform are a *matched pair* inherited from upstream.  Changing one without
      the other rescales the effective filter gain by a factor of L.
    * ``k.split(k.shape[1] // 2, dim=1)`` splits the *time* axis (the filter has been
      squeezed to ``[D, L]``), not the channel axis.
    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    k_f = adjust_filter_shape_for_broadcast(u, k_f)
    k = k.squeeze()

    if bidirectional:
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
        k, k2 = k.split(k.shape[1] // 2, dim=1)
        k2_f = torch.fft.rfft(k2, n=fft_size) / fft_size
        y1 = u_f * k_f
        y2 = u_f.conj() * k2_f.conj()

        y = torch.fft.irfft(y1 + y2, n=fft_size, norm="forward")[..., :seqlen]

    else:
        if k_rev is not None:
            k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
            k_f = k_f + k_rev_f.conj()

        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
        
        y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    # Learnable feed-through (skip) term: the long filter already implies a skip
    # path, and making it explicit lets each channel learn how much of the raw
    # signal to pass through un-filtered.

    return out.to(dtype=u.dtype)


def canonicalize_modal_system(poles, residues):
    """Canonicalize a modal system.

    Args:
        poles (Tensor): The poles of the system.
        residues (Tensor): The residues of the system.

    Returns:
        Tuple[Tensor, Tensor]: The canonicalized poles and residues.
    """
    # Unimplemented in this vendored copy: no caller in eScreen reaches it (the
    # canonical (real) form is never used; the modal form is used instead).
    raise NotImplementedError


def list_tensors(idx):
    """Dump every live torch tensor in the process to ``tensors_<idx>.txt``.

    Parameters
    ----------
    idx : int
        Suffix of the output file (appended on every call).

    Returns
    -------
    None

    Notes
    -----
    Debug helper only: it walks ``gc.get_objects()``, which is extremely slow and
    unsafe while training (it touches unrelated live objects).  Nothing in the
    model calls it.
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and isinstance(obj, torch.Tensor):
                # dump to log
                #print(type(obj), obj.size())
                el = obj[0]
                with open(f"tensors_{idx}.txt", "a") as f:
                    f.write(f"{type(obj)} {obj.size()} {el}\n")
        except Exception:
            pass


class HyenaInferenceEngine:
    """Dispatcher for the FIR/IIR kernels of one Hyena layer.

    One instance is created per ``HyenaCascade``.  The object holds no tensors --
    only the layer index and a handful of switches -- and exposes:

    * ``parallel_fir`` / ``parallel_iir`` -- whole-sequence (training) computation;
    * ``step_fir`` / ``step_iir`` -- one-token recurrence used for decoding;
    * ``prefill_via_*`` -- state-cache construction (most raise
      ``NotImplementedError``);
    * ``_compute_state`` -- modal state for a whole sequence.

    Parameters
    ----------
    fir_fn : callable, optional
        Short-filter convolution.  Stored as ``self.fir_fn`` but never read: every
        method receives its convolution function as an argument.
    iir_prefill_style : str, optional
        Must be one of ``IIR_PREFILL_MODES`` (asserted in the body); default
        ``"modal-fft"``.
    layer_idx : int, optional
        Block index, used as the key into the FIR/IIR state dictionaries.
    ground_truth_activations_path : str, optional
        Stored and never read -- a debugging leftover from the upstream repo.
    hyena_flip_x1x2 : bool, optional
        Swap the roles of ``x1`` and ``x2`` inside the gated branches.
    bidirectional : bool, optional
        Accepted but **not stored**; each method takes its own ``bidirectional``
        argument, which is the value ``HyenaCascade`` passes down.

    Notes
    -----
    ``low_mem_mode`` is hard-coded to ``False`` and only consulted at the end of
    ``parallel_iir``.  Nothing here is stateful across calls except through the
    ``inference_params`` object the caller owns.
    """
    def __init__(
        self,
        fir_fn=None,
        iir_prefill_style="modal-fft",
        layer_idx=None,
        ground_truth_activations_path=None,
        hyena_flip_x1x2=False,
        bidirectional=True
    ) -> None:
        """Store the switches; no tensors or buffers are allocated."""
        self.fir_fn = fir_fn
        assert iir_prefill_style in IIR_PREFILL_MODES, f"iir_prefill_style must be one of {IIR_PREFILL_MODES}"
        self.iir_prefill_style = iir_prefill_style
        self.layer_idx = layer_idx
        self.low_mem_mode = False
        self.ground_truth_activations_path = ground_truth_activations_path
        self.hyena_flip_x1x2 = hyena_flip_x1x2

    def parallel_fir(
        self,
        fir_fn,
        u,
        weight,
        bias,
        L,
        dims,
        groups=None,
        gated_bias=False,
        column_split_hyena=False,
        dim_last=True,
        fir_length=3,
        gate=False,
        inference_params=None,
        prefill_mode=None,
        padding_mask=None,
        bidirectional=True
    ):
        """Apply a depth-wise FIR filter over the whole sequence.

        Parameters
        ----------
        fir_fn : callable
            ``torch.nn.functional.conv1d`` or a ``FlashDepthWiseConv1d`` instance.
            The code branches on identity (``fir_fn != F.conv1d``) to pick the
            flash path.
        u : torch.Tensor
            ``[B, L, C]`` when ``dim_last`` else ``[B, C, L]``.
        weight : torch.Tensor
            ``[C, 1, L_fir]`` depth-wise kernels (``C = 3D`` for the short filter,
            ``D`` for the inner filter).
        bias : torch.Tensor or None
            Per-channel term; added, or multiplied into ``u`` when ``gated_bias``.
        L : int
            Ignored -- the length is re-read from ``u`` on the first line.
        dims : tuple
            ``(hidden_size, num_attention_heads, head_dim, state_size, groups)``.
        groups : int, optional
            Number of filter groups; used only on the FFT branch.  The conv1d branch
            always passes ``groups=u.shape[1]``, i.e. strict depth-wise convolution.
        gated_bias : bool, optional
            If True the bias enters multiplicatively (``z += bias * u``).
        column_split_hyena : bool, optional
            Reinterpret the channel axis as heads before splitting x2/x1/v.
        dim_last : bool, optional
            Layout flag for ``u``.
        fir_length : int, optional
            Kernel length; ``>= 128`` selects the FFT path, otherwise conv1d.
        gate : bool, optional
            If True, split ``u`` into x2/x1/v, convolve ``x1 * v`` and gate by x2.
        inference_params : optional
            When given, the trailing ``fir_length - 1`` samples of ``u`` are returned
            as the cache state.
        prefill_mode : optional
            Unused.
        padding_mask : torch.Tensor, optional
            ``[B, L]`` mask applied to the output.
        bidirectional : bool, optional
            Only honoured on the FFT branch (see Notes).

        Returns
        -------
        tuple
            ``(z, fir_state)``; ``z`` keeps the input layout and dtype, and
            ``fir_state`` is ``None`` unless ``inference_params`` was supplied.

        Notes
        -----
        * ``padding=fir_length - 1`` plus ``[..., :L]`` makes the conv1d branch
          **causal**: output position t only reads inputs ``<= t``.  Consequently
          HyenaSE's length-7 inner filter behaves causally even though the block is
          configured with ``bidirectional=True`` -- that flag only affects the FFT
          branch below.  This is the one place where the code is narrower than the
          manuscript's "bidirectional, no causal mask" description.
        * All arithmetic runs in float32 and is cast back to ``u.dtype``, so the
          bf16 model does not accumulate filter rounding in bf16.
        * ``gated_bias`` turns the "bias" into a learned per-channel gain on the
          input, which is how the Hyena skip term is expressed for long filters.
        """
        L = u.shape[1] if dim_last else u.shape[2]
        if gate:
            hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
            # Compatibility with training infra that column splits the projections
            if column_split_hyena:
                x2, x1, v = column_split(u, num_attention_heads, hidden_size_per_attention_head)
            else:
                x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
            if self.hyena_flip_x1x2:
                x1, x2 = x2, x1
            u = x1 * v

        # prepare input layout, dimensions and dispatch to fir kernel
        # Deprecated
        if fir_fn != torch.nn.functional.conv1d:
            if dim_last:
                u = u.permute(0, 2, 1)  # B, D, L
            z = fir_fn(u)[:, :L]  # B, L, D

        elif fir_length >= 128:
            # Long-filter branch: an FFT convolution over the *linear* convolution
            # window.  This is the only branch that honours `bidirectional`, and it is
            # the path taken by HyenaMR (fir_inner_filter_length = 128).
            with torch.autocast("cuda"):
                z = fftconv_func(
                    u.to(torch.float32),
                    weight[:, :, :L].to(torch.float32),
                    bias,
                    None,
                    gelu=False,
                    bidirectional=bidirectional,
                    groups=groups,
                    layer_idx=self.layer_idx,
                )
                z = z.to(u.dtype)
        else:
            if dim_last:
                u = u.permute(0, 2, 1)  # B, D, L

            if groups is None:
                g = u.shape[1]
            else:
                g = groups
            
            #print(weight.shape)
            z = fir_fn(
                u.to(torch.float32),
                weight.to(torch.float32),
                bias=None,
                stride=1,
                padding=fir_length - 1,
                groups=u.shape[1],  # always set to D, regardless of filter grouping
            # NOTE: `groups` is hard-coded to u.shape[1] here, so the conv1d branch is
            # always strictly depth-wise; the `groups` argument from the caller only
            # takes effect on the FFT branch above.  The `g` local computed on the
            # lines above is therefore unused.
            )[..., :L]

            z = z.to(u.dtype)

            if bias is not None:
                if gated_bias:
                    z = z + bias[None, :, None] * u
                else:
                    z = z + bias[None, :, None]
            # With gated_bias the bias multiplies the input instead of being added:
            # this is the Hyena skip term (D * x) expressed through the same code
            # path as a plain bias, which is why D has shape [D] and not [1].

        # handle padding post fir, the only place with biases
        if type(padding_mask) == torch.Tensor:
            z = z * padding_mask[:, None]

        if gate:
            z = x2 * z

        if inference_params is not None:
            fir_state = u[..., -fir_length + 1 :]
        else:
            fir_state = None

        return z, fir_state

    def parallel_iir(
        self,
        z_pre,
        h,
        D,
        L,
        poles,
        residues,
        t,
        dims,
        layer_idx,
        inference_params=None,
        prefill_style="fft",
        fftconv_fn=None,
        padding_mask=None,
        use_flashfft=False,
        column_split_hyena=False,
        long_fir_threshold=None,
    ):
        """Compute the output state of the short convolutional filter."""
        # Parameters (the one-line summary above is kept; details here):
        #   z_pre          [B, 3D, L]   output of the short FIR filter
        #   h              [1, D, L]    long filter (impulse response), or None
        #   D              [D]          learnable feed-through gain
        #   L              int          sequence length
        #   poles          [G, S, 1]    log-poles of the modal system (only used by
        #                               the prefill/step paths, not in the FFT path)
        #   residues       [G, S]       modal residues (likewise unused here)
        #   t              unused here (only prefill_via_modal_fft needs positions)
        #   dims           5-tuple      (D, H, D/H, state_size, groups)
        #   prefill_style / use_flashfft / fftconv_fn / column_split_hyena /
        #   long_fir_threshold are read only by the optional branches below; with
        #   eScreen's settings (inference_params=None, use_flashfft=False,
        #   long_fir_threshold=None) they are inert.
        #
        # Returns [B, L, D]: the gated result transposed back to channel-last.
        fft_size = 2 * L
        hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
        # Compatibility with training infra that column splits the projections
        if column_split_hyena:
            z = z_pre.reshape(
                z_pre.shape[0],
                num_attention_heads,
                3 * hidden_size_per_attention_head,
                z_pre.shape[2],
            )
            x2, x1, v = (
                z[:, :, :hidden_size_per_attention_head],
                z[
                    :,
                    :,
                    hidden_size_per_attention_head : 2 * hidden_size_per_attention_head,
                ],
                z[:, :, 2 * hidden_size_per_attention_head :],
            )
            x2, x1, v = (
                x2.reshape(x2.shape[0], -1, x2.shape[-1]),
                x1.reshape(x1.shape[0], -1, x1.shape[-1]),
                v.reshape(v.shape[0], -1, v.shape[-1]),
            )
        else:
            x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)

        # The x2/x1/v split mirrors HyenaCascade's projection order (see
        # column_split above).  x1 * v is the "input" of the long filter and x2 is the
        # gate applied afterwards.
        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        x1v = x1 * v

        if inference_params is not None and prefill_style == "recurrence":
            y = self.prefill_via_direct_recurrence(
                inference_params=inference_params,
                x1v=x1v,
                L=L,
                poles=poles,
                residues=residues,
            )

        else:
            if use_flashfft and (L % 2) == 0:  # only works with even L
                y = fftconv_fn(
                    x1v.to(dtype=torch.bfloat16).contiguous(),
                    h.to(dtype=torch.float32),
                )
                X_s = None

            elif long_fir_threshold is None:
                H = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
                X_s = torch.fft.fft(x1v.to(dtype=torch.float32), n=fft_size)
                X = X_s[..., : H.shape[-1]]
                if len(z_pre.shape) > 3:
                    H = H.unsqueeze(1)
                y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]

            else:
                assert h.shape[0] == 1, "batch size must be 1 for long_fir_threshold"
                h = h[0][:, None]  # rearrange to d, 1, l for depthwise conv1d
                h = h[..., :long_fir_threshold]
                y = F.conv1d(
                    x1v,
                    h.to(dtype=x1v.dtype),
                    stride=1,
                    groups=x1v.shape[1],
                    padding=h.shape[-1] - 1,
                )[..., :L]

        y = y.to(dtype=x1v.dtype)
        y = (y + x1v * D.unsqueeze(-1)) * x2
        # Gated output: long_filter(x1 * v) + D * (x1 * v), then multiplied
        # element-wise by x2.  The multiplication happens here, not inside the
        # convolution, so it stays cheap in whatever dtype the activations are.

        if inference_params is not None:
            if prefill_style == "fft":
                self.prefill_via_modal_fft(
                    inference_params=inference_params,
                    x1v=x1v,
                    X_s=X_s,
                    L=L,
                    t=t,
                    poles=poles,
                    dims=dims,
                    layer_idx=layer_idx,
                    use_flashfft=use_flashfft,
                    fftconv_fn=fftconv_fn,
                )

            elif prefill_style == "recurrence":
                # recurrent prefill is done before
                pass
            else:
                raise NotImplementedError
            if self.low_mem_mode:
                # TODO: smarter gc
                del z_pre, x2, x1, v, x1v, h, poles, residues
                torch.cuda.empty_cache()

        return y.permute(0, 2, 1)
        # (B, D, L) -> (B, L, D) so the caller can feed a Linear layer directly.

    def step_fir(self, u, fir_state, weight, bias=None, gated_bias=False, flip_filter=False):
        """Steps forward FIR filters in the architecture.

        FIR filters generally include truncated convolutions in Hyena with an explicit or hybrid time-domain parametrization:
        * Short FIR filters in Hyena featurizers
        * Short and medium FIR filters in Hyena operators

        Note:
            `fir_state` contains the last FIR filter length - 1 elements of `u`: `u_(L-2), u_{L-1), ...`
            We assume dimensions of `short_filter_weight` to be `[d, 1, short_filter_len]`.
        """
        # Only the last `fir_length - 1` samples are needed as the recurrent state:
        # the next step's window is these samples followed by the new token.
        weight = weight.squeeze()

        cache_size = fir_state.shape[-1]
        filter_length = weight.shape[-1]
        if flip_filter:
            weight = weight.flip(-1)
            weight = weight[..., -cache_size - 1 :].unsqueeze(0)
        else:
            weight = weight[..., : cache_size + 1].unsqueeze(0)

        input_dtype = u.dtype
        weight = weight.to(torch.float32)
        u = u.to(torch.float32)
        fir_state = fir_state.to(torch.float32)
        bias = bias.to(torch.float32) if bias is not None else None

        h0, h = weight[..., -1], weight[..., :-1]
        y = h0 * u + torch.sum(fir_state * h, dim=-1)

        if bias is not None:
            if gated_bias:
                y = y + bias * u
            else:
                y = y + bias

        # Update the state
        if cache_size < filter_length - 1:
            fir_state = torch.cat([fir_state, u[..., None]], dim=-1)
        else:
            fir_state = torch.roll(fir_state, -1, dims=2)
            fir_state[..., -1] = u

        return y.to(input_dtype), fir_state

    def step_iir(self, x2, x1, v, D, residues, poles, iir_state, iir_groups=1):
        """Advance the modal IIR filter by one token.

        Parameters
        ----------
        x2 : torch.Tensor
            ``[B, D]`` gate.
        x1 : torch.Tensor
            ``[B, D]`` input of the filter.
        v : torch.Tensor
            ``[B, D]`` value branch.
        D : torch.Tensor
            ``[D]`` feed-through gain.
        residues : torch.Tensor
            ``[G, S]`` modal residues (float32).
        poles : torch.Tensor
            ``[G, S, 1]`` **log**-poles (float32).
        iir_state : torch.Tensor
            ``[B, G, S]`` recurrent state, updated and returned.
        iir_groups : int, optional
            Only ``1`` is implemented.

        Returns
        -------
        tuple
            ``(y, iir_state)`` with ``y`` of shape ``[B, D]``.

        Notes
        -----
        The recurrence is ``state <- exp(pole) * state + x1 * v`` and the output is
        ``(residues . state + D * x1 * v) * x2`` -- the discrete-time form of the
        continuous filter ``sum_k r_k exp(p_k t)``, i.e. exactly the filter that
        ``HyenaCascade.compute_filter`` materialises in the parallel path.
        """
        # TODO: kernelize
        x1v = x1 * v
        poles = torch.exp(poles)  # poles arg contains log_poles
        # exp() converts the stored log-poles back to poles; keeping them in log
        # space during training guarantees a positive (stable) decay factor.
        poles = poles[..., 0][None]  # squeeze dummy seqlen dim and add dummy batch dim
        residues = residues[None]  # add dummy batch dim
        iir_state = poles * iir_state + x1v[..., None]

        res_state = torch.sum(residues * iir_state, dim=-1)

        if iir_groups > 1:
            raise NotImplementedError
        # if self.layer_idx == 2:
        #    breakpoint()
        y = x2 * (res_state + D * x1v)

        return y, iir_state

    def prefill_via_fir_caching(self, u, inference_params, L, *args, **kwargs):
        """Turns the IIR filter into a FIR and uses a cache for decoding."""
        raise NotImplementedError(":)")

    def prefill_via_direct_recurrence(self, inference_params, x1v, L, residues, poles, *args, **kwargs) -> torch.Tensor:
        """
        Compute the IIR state via explicit recurrence (modal form)

        This is the most memory efficient prefilling method for Hyena filters.

        Note:
            dtypes: [state: float32, poles: float32, x1v: bfloat16, output: bfloat16]
        """
        # Note the dtype contract of the upstream docstring: state/poles float32,
        # x1v and the returned output bfloat16.  The interleaved real/imaginary
        # state (``[..., sdim, 2]``) is how the original complex recurrence is
        # carried without complex tensors.
        state_dim = poles.shape[1]
        x1v_ = x1v[..., None, None]  # b, d, l, sdim, reim
        x1v_ = x1v_.repeat(1, 1, 1, state_dim, 2)  # b, d, l, sdim, reim
        x1v_[..., 1] = 0

        state = 0 * x1v_[:, :, 0]
        output = 0 * x1v_[:, :, :, 0, 0]  # b, d, l

        # suppress dummy seqlen dimension
        poles = poles[:, :, 0][None]
        residues = residues[:, :, 0][None].repeat(x1v_.shape[0], 1, 1, 1)  # b, d, sdim, reim

        # state: b, d, sdim, reim
        # poles: 1, d, sdim, reim
        # x1v_: b, d, l, sdim, reim
        for i in range(L):
            state[..., 0] = poles[..., 0] * state[..., 0] - poles[..., 1] * state[..., 1] + x1v_[:, :, i, :, 0]
            state[..., 1] = poles[..., 0] * state[..., 1] + poles[..., 1] * state[..., 0] + x1v_[:, :, i, :, 1]
            output[:, :, i] = torch.sum(residues * state, dim=-2)[..., 0]  # .real

        inference_params.state_dict[self.layer_idx] = state.to(dtype=torch.float32)

        return output

    def prefill_via_hybrid_recurrence(self, inference_params, u, log_poles, x1v_f_a, L, *args, **kwargs):
        """
        Compute the IIR state via hybrid recurrence-convolution over blocks
        """
        raise NotImplementedError(":)")
        # Blocked recurrence/convolution hybrid: never implemented here, and
        # unreachable from eScreen, which never prefills a cache.

    def prefill_via_scan(self, u, inference_params=None, *args, **kwargs):
        """Parallel-scan prefill of the IIR state (not implemented in this copy).

        Parameters
        ----------
        u : torch.Tensor
            Filter input.
        inference_params : optional
            Target state holder.
        *args, **kwargs
            Ignored.

        Returns
        -------
        None
            Always raises NotImplementedError.
        """
        raise NotImplementedError

    def prefill_via_canonical_fft(self, u, inference_params=None, *args, **kwargs):
        """
        Compute the IIR state via a single FFT

        This is the most memory efficient "parallelized" prefilling method for Hyena.

        From: https://arxiv.org/abs/2310.18780
        """
        raise NotImplementedError(":)")
        # Unimplemented: the canonical (real-valued) FFT prefill described in
        # arXiv:2310.18780 was not ported.  The modal variant below is the one that
        # exists; neither is used by eScreen.

    def prefill_via_modal_fft(
        self,
        inference_params,
        x1v,
        L,
        poles,
        t,
        dims,
        layer_idx,
        X_s=None,
        use_flashfft=False,
        fftconv_fn=None,
        state_dtype=torch.float32,
        *args,
        **kwargs,
    ):
        """
        Compute the IIR state via a single FFT
        """
        # Implementation note: this is the "modal-fft" prefill -- the whole input is
        # transformed once (X_s, already computed by parallel_iir) and multiplied by
        # the FFT of exp(poles * t), whose value at index L-1 is exactly the final
        # state of the recurrence.  It is O(L log L) instead of the O(L) recurrence,
        # but runs entirely in parallel on the GPU.
        # When the model has a long convolution derived from a recurrence in modal form and prefill_style is "fft",
        # we split the filter into poles and residues and reuse FFT computation on the input.
        hidden_size, _, _, state_size, hyena_filter_groups = dims

        assert X_s is not None
        bs = x1v.shape[0]
        fft_size = 2 * L
        # poles = torch.view_as_complex(poles.to(torch.float32))
        state_s = (poles.to(torch.float32) * t).exp()

        # state_s = poles**t
        state_S = torch.fft.fft(state_s, n=fft_size).repeat(bs, 1, 1, 1)  # B, D, state_dim, 2 * L
        if hyena_filter_groups > 1:
            state_S = state_S.repeat_interleave(hidden_size // hyena_filter_groups, 1)
        state = torch.fft.ifft(X_s[..., None, :] * state_S, n=fft_size)
        inference_params.state_dict[layer_idx] = state[..., L - 1].to(dtype=state_dtype)

    def _compute_state(self, log_poles, u, t, L, *args, **kwargs):
        """
        Compute the IIR state given an input `u` and log_poles of the modal system.
        """
        # Same modal trick as prefill_via_modal_fft, but kept for callers that only
        # want the state (returns the whole state sequence, not just its last row).
        # U is computed with rfft while X uses the full fft: the product is valid
        # because irfft/fft here are applied to the same length-2L grid.
        bs = u.shape[0]
        fft_size = 2 * L
        U = torch.fft.rfft(u.to(torch.float32), n=fft_size)
        fft_size = 2 * L
        x = (log_poles * t).exp()
        # [batch, hidden_size, state_dim, 2 * seqlen]
        X = torch.fft.fft(x, n=fft_size).repeat(bs, 1, 1, 1)
        state = torch.fft.ifft(U[..., None, :] * X, n=fft_size)[..., :L]
        return state


class HyenaFilter:
    """Handles Hyena filter computations including FFT and direct convolution."""
    # NOTE: this class is incomplete in this repository -- the file ends after
    # _prepare_filter, so `fft_conv` calls a `_compute_fft_conv` method that is never
    # defined (AttributeError if used).  Nothing in eScreen instantiates
    # HyenaFilter; the convolution actually used is fftconv_func above.

    def __init__(self, use_flash_fft=False):
        """Store the "use flash fft" switch (read nowhere in this class)."""
        self.use_flash_fft = use_flash_fft

    def fft_conv(self, u, k, D, **kwargs):
        """FFT-based convolution implementation."""
        # Adds the feed-through term u * D to the FFT convolution result; note that
        # self._compute_fft_conv does not exist (see the class note above).
        seqlen = u.shape[-1]
        fft_size = 2 * seqlen

        k_f = self._prepare_filter(k, u, fft_size)
        y = self._compute_fft_conv(u, k_f, fft_size, seqlen, **kwargs)

        return y + u * D.unsqueeze(-1)

    def _prepare_filter(self, k, u, fft_size):
        """Prepare filter for FFT convolution."""
        # Pre-divides the kernel spectrum by fft_size so that the inverse transform
        # (which upstream applies with norm="forward") yields the unscaled
        # convolution; see the note in fftconv_func for why the two factors are a
        # matched pair.
        k_f = torch.fft.rfft(k, n=fft_size) / fft_size
        return adjust_filter_shape_for_broadcast(u, k_f)