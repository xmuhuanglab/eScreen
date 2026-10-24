"""eScreen: sequence-based prediction and dissection of CRE activity.

Package layout, mapped to the Supplementary Methods:

===================  =======================================================
module               role
===================  =======================================================
``model``            full eScreen model + nucleotide-level attribution
                     (Sections 2.1-2.3, 3.2)
``escreen``          the same model without the ``captum`` attribution path
``CellEmb``          Geneformer-based cell-type embeddings (Section 2.3.5)
``motif``,           PWM loading and motif pre-processing (Section 2.3.6)
``motif_tool``
``genome_tool``      sequence retrieval, one-hot encoding, interval
                     arithmetic, locus plotting (Sections 2.3.1, 2.4.3)
``miaomiao_tool``    peak/summit handling and signal normalisation
                     (Sections 1.3, 2.4.2)
``utils``            class-balanced sampler, losses and metrics
                     (Sections 2.3.2-2.3.4, 7.1)
``serve``            Streamlit front end (predict / attribute / train)
``SH2core``          StripedHyena2 backbone blocks (Section 2.1)
===================  =======================================================

Importing the package pulls in the model, which needs PyTorch, and (through
``model``) ``captum`` and ``transformers``.
"""

from escreen.model import eSCREEN_backbone, eScreen_vX
from escreen import CellEmb
from escreen import genome_tool
from escreen import miaomiao_tool
from escreen import motif_tool
from escreen import motif
from escreen import utils
from escreen import SH2core

__all__ = [
    'eSCREEN_backbone',
    'eScreen_vX',
    'CellEmb',
    'genome_tool',
    'miaomiao_tool',
    'motif_tool',
    'motif',
    'utils',
    'SH2core',
]