"""Parallel gated MLP ("GLU-style" feed-forward) used inside every eScreen block.

Provides :func:`grab_first_if_tuple` and :class:`ParallelGatedMLP`.  The block is
the feed-forward half of both the Hyena blocks (``ParallelGatedConvBlock``) and
the attention block (``AttentionBlock``), i.e. it is instantiated once per
StripedHyena2 block in the four-block eScreen backbone
(Supplementary Methods §2.1).

The operation is the gated MLP introduced for Hyena/Evo-style architectures::

    z1 = W1 z            [B, L, inner]
    z2 = W2 z            [B, L, inner]
    y  = W3 ( act(z1) * z2 )   [B, L, D]

i.e. two independent up-projections whose element-wise product is gated by an
activation (or, for layers with ``layer_idx > 0`` under
``evo2_style_activations``, not gated by any non-linearity at all) and then
projected back down.  Note the multiplicative interaction: this is a GLU, not a
plain two-layer MLP.

eScreen configuration (``escreen/escreen.py``, ``get_block``): ``hidden_size`` =
``d_model`` (128 distilled / 256 full), ``inner_mlp_size = 256`` and
``inner_size_multiple_of = 16``, ``mlp_activation = 'gelu'``,
``evo2_style_activations = True``, ``model_parallel_size = 1``.

Adapted from the StripedHyena2 / Evo2 reference implementation.
"""

import torch.nn as nn
import torch.nn.functional as F
from typing import Optional





def grab_first_if_tuple(x):
    """Return ``x[0]`` if ``x`` is a tuple, else ``x`` unchanged.

    Parameters
    ----------
    x : Any
        Usually a tensor, but may be the ``(output, residual)`` tuple returned by
        fused-dense projections when ``return_residual=True``.

    Returns
    -------
    Any
        The tensor payload.

    Notes
    -----
    The check is done on the *class name* rather than with ``isinstance`` to
    avoid importing/ depending on the tuple type of the fused kernel.  With the
    plain ``nn.Linear`` layers used in eScreen this function is an identity, so
    the whole mechanism is inert here and only exists for checkpoint/backend
    compatibility.
    """
    if x.__class__.__name__ == "tuple":
        return x[0]
    else:
        return x

class ParallelGatedMLP(nn.Module):
    """Parallel (gated) feed-forward network.

    Computes ``y = W3(act(W1 z) * W2 z)`` with shapes
    ``[B, L, D] -> [B, L, inner] -> [B, L, D]``, where
    ``inner = inner_mlp_size`` when given, otherwise
    ``ceil(8 * D / 3 / multiple_of) * multiple_of``.

    Parameters
    ----------
    hidden_size : int
        Model width ``D`` (128 distilled / 256 full, §2.1).
    layer_idx : int
        Index of the block this MLP belongs to (0..3 in eScreen).  Used *only*
        to decide whether ``evo2_style_activations`` disables the activation.
    inner_size_multiple_of : int, optional
        Round the computed inner width up to a multiple of this value, times
        ``model_parallel_size`` (default 64; eScreen passes 16).
    mlp_activation : {'gelu', 'silu'}, optional
        Activation applied to the first branch (default ``'gelu'``).
    evo2_style_activations : bool, optional
        If True, blocks with ``layer_idx > 0`` replace the activation by
        ``nn.Identity()`` (default False; eScreen sets True).
    model_parallel_size : int, optional
        Tensor-parallel degree used only to widen ``multiple_of`` (default 1).
    inner_mlp_size : int or None, optional
        Explicit inner width; when given it overrides the heuristic entirely
        (eScreen passes 256).

    Notes
    -----
    All three projections are bias-free (``bias=False``), so the parameter
    layout has no ``.bias`` keys and matches the upstream checkpoints.
    """
    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        inner_size_multiple_of: int = 64,
        mlp_activation: str = "gelu",
        evo2_style_activations: bool = False,
        model_parallel_size: int = 1,
        inner_mlp_size: Optional[int] = None,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        # ``layer_idx`` takes part in no computation here; it is kept because the
        # Evo2 activation convention (see below) is defined per layer index.
        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError(f"Activation function {self.act_type} not implemented")
        
        if self.layer_idx > 0 and evo2_style_activations:
            self.act = nn.Identity()
            # Evo2 convention applied by eScreen (``evo2_style_activations=True``):
            # every MLP except the first block's is *linear* -- the gate becomes
            # act(z1) * z2 == z1 * z2.  In eScreen this affects the blocks at
            # layer_idx 1, 2 and 3 (HyenaMR, HyenaLI, Attention) and leaves only
            # block 0 (HyenaSE) with a GELU gate.

        self.multiple_of = multiple_of * model_parallel_size

        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * ((inner_size + self.multiple_of - 1) // self.multiple_of)
        # Standard transformer FFN width 4D, reduced to 8D/3 because the gated
        # MLP carries two up-projections (2 * W1-style matrices) instead of one,
        # i.e. 8/3 * D keeps the parameter count comparable to a 4D MLP.  The
        # result is then rounded UP to a multiple of ``multiple_of`` so that the
        # width is divisible by the tensor-parallel shard / vectorisation unit.
        inner_size = inner_mlp_size if inner_mlp_size is not None else inner_size
        # ``inner_mlp_size`` (256 in every eScreen block) discards the value
        # computed above, which means ``inner_size_multiple_of`` has no effect on
        # the eScreen backbone.  Both branches are kept for config compatibility.

        self.l1 = nn.Linear(
        # Two parallel up-projections (the "gate" pair) plus one down-projection.
        # No biases: see the class Notes.
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        """Apply the gated feed-forward transformation.

        Parameters
        ----------
        z : torch.Tensor
            Input of shape ``[B, L, hidden_size]`` (bfloat16 in eScreen; the
            block's post-norm output).

        Returns
        -------
        torch.Tensor
            Output of shape ``[B, L, hidden_size]``, same dtype as the weights.

        Notes
        -----
        The multiplication is element-wise *after* the activation, so the second
        branch acts as a per-channel gate rather than as a second additive path.
        ``grab_first_if_tuple`` calls around the intermediate results only matter
        for backends whose Linear returns ``(y, x)``; they are identities here.
        """
        z1, z2 = self.l1(z), self.l2(z)
        z1, z2 = grab_first_if_tuple(z1), grab_first_if_tuple(z2)
        y = self.l3(self.act(z1) * z2)
        # act(z1) * z2 is the GLU gate; with evo2_style_activations and
        # layer_idx > 0, ``act`` is Identity and this is a pure product.
        return grab_first_if_tuple(y)