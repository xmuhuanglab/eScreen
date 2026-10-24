"""StripedHyena2 building blocks used by the eScreen sequence encoder.

Adapted from the StripedHyena2 reference implementation.  eScreen stacks four
of these blocks -- ``HyenaSE``, ``HyenaMR``, ``HyenaLI`` and ``Attention`` -- as
described in Supplementary Methods Section 2.1, using short depth-wise filters
of length 3, a state size of 16, and no causal mask.

The exported classes are:

``ParallelGatedConvBlock``
    Convolutional / state-space block; the ``HyenaSE``, ``HyenaMR`` and
    ``HyenaLI`` variants are all instances of it, differing only in their filter
    arguments.
``AttentionBlock``
    Multi-head self-attention block with a gated MLP.
``ParallelGatedMLP``
    Gated feed-forward network used inside the blocks.
``RMSNorm``
    Root-mean-square layer normalisation.
``HyenaCascade``
    Cascade of Hyena operator variants (used by ``ParallelGatedConvBlock``).
``HyenaInferenceEngine``
    Convolution and FIR primitives, including the bidirectional FFT path.

Note that ``HyenaInferenceEngine`` is imported here but omitted from ``__all__``,
so ``from escreen.SH2core import *`` does not re-export it even though the
import itself succeeds.
"""

from escreen.SH2core.AttentionBlock import AttentionBlock
from escreen.SH2core.HyenaCascade import HyenaCascade
from escreen.SH2core.ParallelGatedConvBlock import ParallelGatedConvBlock
from escreen.SH2core.ParallelGatedMLP import ParallelGatedMLP
from escreen.SH2core.RMSNorm import RMSNorm
from escreen.SH2core.HyenaInferenceEngine import HyenaInferenceEngine

__all__ = ["AttentionBlock", "HyenaCascade", "ParallelGatedConvBlock", "ParallelGatedMLP", "RMSNorm"]