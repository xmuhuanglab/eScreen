"""Root-mean-square layer normalisation used throughout the eScreen backbone.

Provides :class:`RMSNorm`, the only normalisation layer used by the four
StripedHyena2 blocks that make up the eScreen sequence encoder
(Supplementary Methods §2.1).  Both the Hyena blocks (``ParallelGatedConvBlock``)
and the attention block (``AttentionBlock``) instantiate two of these each -- a
*pre*-norm on the input to the residual branch and a *post*-norm feeding the
gated MLP -- so this class is the most frequently executed normaliser in the
model.

The layer has a single learnable per-channel gain (no bias, no mean
subtraction), which keeps it cheap enough to run in bfloat16 inside all four
blocks.  eScreen builds it in the default dtype and then casts whole blocks with
``.to(torch.bfloat16)`` (see ``get_block`` in ``escreen/escreen.py``), so the
gain ends up in bfloat16 in the deployed model.

Adapted from the StripedHyena2 reference implementation.
"""

import torch

class RMSNorm(torch.nn.Module):
    """Root-mean-square normalisation with a learnable per-channel gain.

    For each row ``x`` of length ``D = hidden_size`` the layer computes

        y = x / (||x||_2 / sqrt(D) + eps) * scale

    where ``||x||_2`` is the L2 norm taken over the last dimension and ``scale``
    is a learnable vector of ones (shape ``[D]``).

    Parameters
    ----------
    hidden_size : int
        Size ``D`` of the normalised (last) axis.  ``D = 128`` in the distilled
        backbone and ``D = 256`` in the full task configuration (§2.1).
    eps : float, optional
        Floor added to the root-mean-square for numerical safety
        (default ``1e-6``).
    use_flash_rmsnorm : bool, optional
        If True, dispatch to ``flash_attn.ops.rms_norm.rms_norm`` instead of the
        explicit formulation below (default False; eScreen always uses the
        explicit path, so the flash branch is dead code here).

    Notes
    -----
    Two details of this vendored implementation differ from the textbook
    RMSNorm and should not be "fixed" without re-validating the checkpoints:

    1. ``eps`` is added *outside* the square root (``RMS + eps``) rather than
       inside it (``sqrt(mean(x**2) + eps)``).  For any non-degenerate
       activation the two agree to within ``1e-6``.
    2. The norm is divided by ``sqrt(D)`` via the ``hidden_size ** -0.5``
       factor, which turns ``||x||_2`` into the true root mean square so that
       ``y`` has unit RMS before the gain is applied.  Dropping that factor
       would scale every activation by ``sqrt(D)`` (about 11x at ``D = 128``).
    """
    def __init__(self, hidden_size,eps=1e-6,use_flash_rmsnorm=False):
        """Create the gain parameter and optionally bind the flash kernel.

        Parameters
        ----------
        hidden_size : int
            Size ``D`` of the normalised (last) axis; 128 in the distilled
            backbone, 256 in the full task configuration (§2.1).
        eps : float, optional
            Floor added to the root-mean-square, after the square root
            (default ``1e-6``).
        use_flash_rmsnorm : bool, optional
            Eagerly import ``flash_attn.ops.rms_norm`` and store it as
            ``self.rmsnorm_func`` (default False).  eScreen keeps this False, so no
            flash-attn symbol is touched when the layer is built.

        Returns
        -------
        None

        Notes
        -----
        * The gain is ones-initialised, so a freshly built layer is an exact RMS
          normaliser: it cannot disturb the residual stream at step 0.
        * ``self.scale`` is created in the default (float32) dtype; the whole block
          is cast to bfloat16 afterwards in ``escreen/escreen.py``, which is what
          keeps inference in bf16 without silently promoting activations here.
        """
        # expected args: residual eps, hidden size, whether to use fast normalization
        super().__init__()
        self.eps, self.hidden_size = eps, hidden_size
        self.scale = torch.nn.Parameter(torch.ones(hidden_size))
        # Ones-init: the layer starts as a pure RMS normaliser (gain = 1), so at
        # step 0 it cannot distort the residual stream it is applied to.
        self.register_parameter("scale", self.scale)
        # NOTE: ``self.scale`` is already registered by the ``nn.Parameter``
        # assignment on the line above; this second registration is a no-op that
        # is kept for parity with the upstream file.
        self.use_flash_rmsnorm = use_flash_rmsnorm

        if self.use_flash_rmsnorm:
            from flash_attn.ops.rms_norm import rms_norm as rmsnorm_func

            self.rmsnorm_func = rmsnorm_func

    def forward(self, x):
        """Normalise the last dimension, then apply the learnable gain.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[..., D]``; in eScreen ``[B, L, D]`` with
            ``L = 512`` and dtype bfloat16.

        Returns
        -------
        torch.Tensor
            Same shape and dtype as ``x``.

        Notes
        -----
        * The L2 norm is computed in the input dtype (bfloat16 in eScreen), so
          the denominator carries bf16 rounding; ``eps = 1e-6`` is far below
          bf16 resolution and only protects against exactly-zero rows.
        * ``self.scale`` is float32 when the module is constructed and becomes
          bfloat16 only after the caller casts the block; if a bfloat16 input
          is ever fed to an *uncast* module, ``scale * y`` would promote the
          result to float32, which would then fail in the following Linear.
        """
        if self.use_flash_rmsnorm:
            #print('PASS')
            return self.rmsnorm_func(x, self.scale, self.eps)
        else:
            #print('PASS')
            # Explicit formulation: keep the norm in the input dtype and only
            # widen implicitly through ``scale`` (see Notes above).  ``eps`` is
            # deliberately applied after the sqrt -- see the class docstring.
            y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2) + self.eps)
            #print(x.device,y.device,self.scale.device)
            return self.scale * y