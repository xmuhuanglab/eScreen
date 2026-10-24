"""eScreen: genome-wide prediction of cis-regulatory element (CRE) activity.

This module implements the ``eScreen`` branch of the framework described in
Supplementary Methods Sections 2.1-2.3 and Fig. 2d.  The model maps a 512 bp
CRE window together with a cell-type label to a continuous activity score that
is a monotone transform of the MAGeCK RRA value (Section 2.3.2).

Forward pipeline
----------------
::

    one-hot DNA                       [B, 512, 4]
      -> frozen PWM Conv1d            [B, 1274, 512]   Section 2.3.6
      -> learnable linear projection  [B, 512, d_model]
      -> 4 hybrid blocks (residual)   [B, 512, d_model] Section 2.1
      -> mean-pool over length        [B, d_model]
      -> Cell-Conditioned LoRA        [B, d_model]     Section 2.3.5
      -> prototype MoE decoder (K=64) [B, 1]

The four hybrid blocks are, in order, ``HyenaSE``, ``HyenaMR``, ``HyenaLI`` and
a plain multi-head ``Attention`` block; each uses short depth-wise filters of
length 3 and a Hyena state size of 16, and none of them applies a causal mask,
so every position sees both upstream and downstream sequence context
(Section 2.1, "Core Encoder").

Module map
----------
``eSCREEN_backbone``
    The sequence encoder.  Frozen after representation distillation from
    HyenaDNA (Section 2.2) and reused as a fixed feature extractor.
``CellConditionedLoRA``
    Cell-type-conditioned low-rank modulation of the pooled embedding.
``PrototypeDecoder``
    Mixture-of-experts readout that emits the regression value.
``eScreen_vX``
    The full task model: backbone + cell embedding + LoRA + readout heads,
    with the training / inference loops attached.

Model file variant
------------------
This file is the deployable variant of :mod:`escreen.escreen`.  It differs from
that module in exactly two ways:

1. Imports are written package-qualified (``from escreen.utils import ...``) so
   that the module can be imported as ``escreen.model`` rather than only when
   the package directory is on ``sys.path``.
2. It additionally provides :meth:`eScreen_vX.attribute`, which computes
   nucleotide-level Integrated Gradients attributions (Supplementary Methods
   Section 3.2).  That capability pulls in ``captum``, which is why it lives
   here and not in ``escreen.py``.

Apart from this section, the documentation below is shared verbatim with
``escreen.py``; keep the two files in sync when editing either.

Notes
-----
All backbone computation is performed in ``bfloat16`` to match the training
regime reported in Section 2.1; the pooled embedding is cast back to
``float32`` before the readout heads so that the regression head trains stably.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import numpy as np
import os
os.environ['TE']='0'

from escreen.utils import boost_sampler

from escreen.SH2core.ParallelGatedConvBlock import ParallelGatedConvBlock
from escreen.SH2core.AttentionBlock import AttentionBlock
from escreen.SH2core.RMSNorm import RMSNorm
from escreen.SH2core.HyenaInferenceEngine import fftconv_standard

from einops import rearrange
from tqdm import tqdm
from captum.attr import IntegratedGradients

class eSCREEN_backbone(nn.Module):
    """StripedHyena2-based sequence encoder of eScreen (Section 2.1).

    Consumes a one-hot encoded DNA window and returns a single sequence-level
    embedding obtained by averaging the block outputs over the length axis.
    The encoder holds no cell-type information: conditioning happens downstream
    in :class:`CellConditionedLoRA`.

    Parameters
    ----------
    filter_type : {'conv', 'pwm', 'pwm-conv', 'emb'}
        Input-embedding strategy.  The production eScreen model uses ``'pwm'``;
        the other modes are kept for the ablations in Section 5.1.
    d_model : int
        Latent width.  128 for the distilled backbone, 256 for the full
        task-specific configuration (Section 2.1).
    seq_length : int
        512 for eScreen: each ENCODE cCRE is centred and extended by 256 bp on
        both sides (Section 2.3.1).
    """

    def __init__(self,filter_type='conv',d_model=32,num_filters=32,seq_length=500,celltype_num=32,mode='sequence',device='cpu',lr=1e-5,**kwargs):
        """Build the input embedding stage and the four hybrid blocks.

        Parameters
        ----------
        filter_type : {'conv', 'pwm', 'pwm-conv', 'emb'}
            Selects the input embedding (Section 2.1, "Input embedding").

            - ``'conv'``   randomly initialised three-layer Conv1d
            - ``'pwm'``    fixed filters taken from known TF PWMs (default in eScreen)
            - ``'emb'``    plain ``nn.Embedding`` lookup over a 6-token vocabulary
            - ``'pwm-conv'`` reserved, not implemented (falls through to a no-op)
        d_model : int
            Latent width of the encoder.
        seq_length : int
            Maximum sequence length the encoder is expected to see.
        device : str or torch.device
            Stored only; used by :meth:`move`.
        **kwargs
            ``kernel_fwd`` and ``kernel_rev`` are required when
            ``filter_type='pwm'``.  Both must be ``[num_motifs, 4, 35]``
            tensors -- the forward and reverse-complement PWMs stacked as
            convolutional kernels (Section 2.3.6).  ``d_model`` and
            ``layer_idx`` are forwarded to :meth:`get_block`; passing
            ``kernel_fwd`` of shape ``[637, 4, 35]`` yields the documented
            1,274 output channels.
        """
        super().__init__()
        # define the model architecture
        # ------------------------------------------------------------------
        # Input embedding stage (Section 2.1, "Input embedding")
        # ------------------------------------------------------------------
        if filter_type == 'conv': # embed with a randomly initialized conv layer
            # 'conv' mode: three stacked Conv1d + ReLU layers, receptive fields
            # of 19 / 11 / 7 bp respectively.
            #
            # NOTE: Section 2.1 documents the channel widths of this mode as
            # 4 -> 256 -> 256 -> d_model.  This implementation instead uses
            # d_model // 2 for the two hidden layers, which halves the parameter
            # count.  The discrepancy does not affect any reported result
            # because the production eScreen model uses 'pwm' mode; it only
            # matters for the input-embedding ablation in Section 5.1.
            padding = 2
            self.conv0 = nn.Sequential(
                nn.Conv1d(in_channels=4, out_channels=d_model//2, kernel_size=19, padding='same'),
                nn.ReLU(),
                nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2, kernel_size=11, padding='same'),
                nn.ReLU(),
                nn.Conv1d(in_channels=d_model//2, out_channels=d_model, kernel_size=7, padding='same'),
            ).to(torch.bfloat16)
            self.motif_embedding = nn.Linear(d_model,d_model).to(torch.bfloat16) # embedding that encodes motifs
        elif filter_type == 'pwm': # embed using motif PWMs
            # 'pwm' mode -- the production eScreen input stage (Section 2.3.6).
            #
            # 637 PWMs from Vierstra et al. are each padded to a (4, 35) matrix
            # and stacked as the weights of a single Conv1d over the one-hot
            # sequence.  The reverse complement of every motif is stacked after
            # the forward strand, giving 1,274 output channels that each
            # correspond deterministically to one (motif, strand) pair.  The
            # convolutional weights are frozen so that a channel's activation
            # remains interpretable as the binding evidence of one known TF
            # motif; only the following linear projection is trained.
            #
            # Both strand tensors are supplied by the caller: ``kernel_fwd``
            # holds the forward PWMs and ``kernel_rev`` their reverse
            # complements, each of shape ``[num_motifs, 4, kernel_size]``.
            # Concatenating along dim 0 pairs motif ``i`` on the forward strand
            # with channel ``i + num_motifs`` on the reverse strand, so a single
            # motif always occupies a fixed pair of channels and downstream motif
            # attribution can map a channel index back to (motif, strand).
            kernel_fwd = kwargs.get('kernel_fwd')
            kernel_rev = kwargs.get('kernel_rev')
            kernel_size = kernel_fwd.shape[-1]
            # compute the deterministic padding: padding = (kernel_size - 1) // 2
            # (recorded for reference only -- the conv below uses 'same' padding)
            padding = (kernel_size - 1) // 2
            self.conv0 = nn.Conv1d(in_channels=4, out_channels=2*len(kernel_fwd), kernel_size=kernel_fwd.shape[-1], padding='same')
            # assume kernel_fwd and kernel_rev have shape [num_motifs, 4, kernel_size]
            # concatenate them along dim 0 (the output-channel dim)
            combined_weight = torch.cat([kernel_fwd, kernel_rev], dim=0)
            self.conv0.weight.data = combined_weight
            self.conv0.weight.requires_grad = False
            self.conv0 = self.conv0.to(torch.bfloat16)
            self.motif_embedding = nn.Linear(2*len(kernel_fwd),d_model).to(torch.bfloat16) # embedding that encodes motifs
        elif filter_type == 'pwm-conv': # PWM hybrid convolution
            pass # not supported yet, skipped
        elif filter_type == 'emb': # use an embedding layer
            self.emb_layer = nn.Embedding(num_embeddings=6, embedding_dim=d_model, padding_idx=0)
            
        self.filter_type = filter_type
        
        self.d_model = d_model # feature dimension
        self.seq_length = seq_length # maximum sequence length
        
        # ------------------------------------------------------------------
        # Core encoder: four hybrid blocks, combined by residual additions
        # (Section 2.1, "Core Encoder")
        #
        # In the StripedHyena2 nomenclature:
        #   HyenaSE   "short explicit" -- a short explicit FIR (inner length 7)
        #             on the state-space path; models fine local motif syntax.
        #   HyenaMR   "multi-resolution" -- the same grouping but an inner FIR
        #             of length 128, widening the receptive field without
        #             stacking more layers.
        #   HyenaLI   "long implicit" -- no explicit FIR at all; relies purely
        #             on the implicit long convolution of the SSM.
        #   Attention  a vanilla 8-head self-attention block that mixes
        #             information globally and complements the convolutional
        #             path.
        #
        # Every block is built with bidirectional=True downstream, i.e. no
        # causal mask, so a position can integrate both upstream and downstream
        # CRE context -- the requirement stated in Section 2.1 for modelling
        # intra-CRE motif combination and sequence syntax.
        # ------------------------------------------------------------------
        self.HyenaCluster=nn.ModuleList([ # cluster of Hyena blocks
            # HyenaSE: filter groups = 128, inner FIR length 7
            self.get_block(
                d_model=d_model,layer_idx=0,num_filters=d_model,num_attention_heads=8,hyena_filter_groups=128,fir_inner_filter_length=7,block_type='HyenaSE',**kwargs
            ),
            # HyenaMR: same grouping, but a much longer inner FIR (128)
            self.get_block(
                d_model=d_model,layer_idx=1,num_filters=d_model,num_attention_heads=8,hyena_filter_groups=128,fir_inner_filter_length=128,block_type='HyenaMR',**kwargs
            ),
            # HyenaLI: no explicit FIR -> implicit long convolution only
            self.get_block(
                d_model=d_model,layer_idx=2,num_filters=d_model,num_attention_heads=8,block_type='HyenaLI',**kwargs
            ),
            # Attention: global mixing, 8 heads
            self.get_block(
                d_model=d_model,layer_idx=3,num_attention_heads=8,block_type='Attention',**kwargs
            ),
        ])

        self.device=device
        
    def get_block(self,block_type='HyenaLI',**kwargs):
        """Instantiate one hybrid block of the core encoder.

        Factory used by :attr:`HyenaCluster`.  It hides the long list of
        StripedHyena2 constructor arguments and pins the values that eScreen
        keeps constant across all four blocks.

        Parameters
        ----------
        block_type : {'HyenaSE', 'HyenaMR', 'HyenaLI', 'Attention'}
            Which block to build.  ``HyenaSE``/``HyenaMR`` require
            ``hyena_filter_groups`` and ``fir_inner_filter_length``;
            ``HyenaLI`` requires neither and therefore falls back to the
            implicit long convolution; ``Attention`` needs no convolution
            arguments at all.
        **kwargs
            Must contain ``d_model``, ``num_filters`` and ``layer_idx`` and,
            for the convolutional variants, ``hyena_filter_groups`` and
            ``fir_inner_filter_length``.

        Returns
        -------
        ParallelGatedConvBlock or AttentionBlock
            The block, cast to ``bfloat16``.

        Notes
        -----
        Values fixed for every block, matching Section 2.1:

        ``state_size = 16``
            Order of the implicit state-space recurrence.
        ``short_filter_length = 3``
            Depth-wise short convolution applied before the long path; 3 bp is
            the shortest kernel that still couples adjacent nucleotides.
        ``bidirectional = True``
            No causal mask, so both strands of context are visible.
        ``use_flashfft = False``, ``use_flash_depthwise = False``
            The FFT fast path is disabled; a 512 bp CRE is short enough that
            the plain recurrence is faster and numerically simpler.
        ``inner_mlp_size = 256``, ``mlp_activation = 'gelu'``
            Shared gated-MLP width and activation.
        """
        if block_type=='HyenaSE':
            return ParallelGatedConvBlock(
                              hidden_size        = kwargs['d_model'],
                              layer_idx          = kwargs['layer_idx'],
                              qkv_proj_bias      = False,
                              hyena_out_proj_bias= True,   
                              state_size  = 16,
                              num_filters = kwargs['num_filters'], # note: this must not exceed hidden_size
                              num_attention_heads= kwargs['num_attention_heads'], # just keep this consistent with the later attention layer
                              short_filter_length= 3,
                              short_filter_bias  = False,
                              hyena_filter_groups    = kwargs['hyena_filter_groups'],     # distinguishes HyenaMR from HyenaSE; must not exceed d_model and must divide d_model
                              fir_inner_filter_length= kwargs['fir_inner_filter_length'], # this parameter is key to distinguishing HyenaMR from HyenaSE
                              inner_size_multiple_of = 16,      # these five parameters belong to the MLP
                              mlp_activation         = 'gelu',  # these five parameters belong to the MLP
                              evo2_style_activations = True,    # these five parameters belong to the MLP
                              model_parallel_size    = 1,       # these five parameters belong to the MLP
                              inner_mlp_size         = 256,     # these five parameters belong to the MLP
                              column_split_hyena     = False, # this one belongs to HyenaCascade
                              interleave             = True,  # this one belongs to HyenaCascade
                              hyena_flip_x1x2        = False, # this one belongs to HyenaCascade
                              use_flash_depthwise    = False, # this one belongs to HyenaCascade
                              use_flashfft           = False, # this one belongs to HyenaCascade
                              prefill_style          = 'fft', # this one belongs to HyenaCascade
                              bidirectional          = True   # this one belongs to HyenaCascade
            ).to(torch.bfloat16)
        elif block_type=='HyenaMR':
            return ParallelGatedConvBlock(
                              hidden_size= kwargs['d_model'],
                              layer_idx  = kwargs['layer_idx'],
                              qkv_proj_bias=False,
                              hyena_out_proj_bias=True,
                              state_size = 16,
                              num_filters= kwargs['num_filters'],
                              num_attention_heads    = kwargs['num_attention_heads'], # just keep this consistent with the later attention layer
                              short_filter_length    = 3,
                              short_filter_bias      = False,
                              hyena_filter_groups    = kwargs['hyena_filter_groups'],     # distinguishes HyenaMR from HyenaSE; must not exceed d_model and must divide d_model
                              fir_inner_filter_length= kwargs['fir_inner_filter_length'], # this parameter is key to distinguishing HyenaMR from HyenaSE
                              inner_size_multiple_of = 16,     # these five parameters belong to the MLP
                              mlp_activation         = 'gelu', # these five parameters belong to the MLP
                              evo2_style_activations = True,   # these five parameters belong to the MLP
                              model_parallel_size    = 1,      # these five parameters belong to the MLP
                              inner_mlp_size         = 256,    # these five parameters belong to the MLP
                              column_split_hyena     = False, # this one belongs to HyenaCascade
                              interleave             = True,  # this one belongs to HyenaCascade
                              hyena_flip_x1x2        = False, # this one belongs to HyenaCascade
                              use_flash_depthwise    = False, # this one belongs to HyenaCascade
                              use_flashfft           = False, # this one belongs to HyenaCascade
                              prefill_style          = 'fft', # this one belongs to HyenaCascade
                              bidirectional          = True   # this one belongs to HyenaCascade
            ).to(torch.bfloat16)
        elif block_type=='HyenaLI':
            return ParallelGatedConvBlock(
                              hidden_size= kwargs['d_model'],
                              layer_idx  = kwargs['layer_idx'],
                              qkv_proj_bias=False,
                              hyena_out_proj_bias=True,
                              state_size = 16,
                              num_filters= kwargs['num_filters'],
                              num_attention_heads= kwargs['num_attention_heads'], # just keep this consistent with the later attention layer
                              short_filter_length= 3,
                              short_filter_bias  = False,
                              proj_groups        = 1,
                              # without hyena_filter_groups and fir_inner_filter_length it becomes HyenaLI
                              # the original config had use_flashfft=False, so the fast-conv module is not passed here either
                              inner_size_multiple_of =16,     # these five parameters belong to the MLP
                              mlp_activation         ='gelu', # these five parameters belong to the MLP
                              evo2_style_activations =True,   # these five parameters belong to the MLP
                              model_parallel_size    =1,      # these five parameters belong to the MLP
                              inner_mlp_size         =256,    # these five parameters belong to the MLP
                              column_split_hyena     = False, # this one belongs to HyenaCascade
                              interleave             = True,  # this one belongs to HyenaCascade
                              hyena_flip_x1x2        = False, # this one belongs to HyenaCascade
                              use_flash_depthwise    = False, # this one belongs to HyenaCascade
                              use_flashfft           = False, # this one belongs to HyenaCascade
                              prefill_style          = 'fft', # this one belongs to HyenaCascade
                              bidirectional          = True   # this one belongs to HyenaCascade
            ).to(torch.bfloat16)
        elif block_type=='Attention':
            return AttentionBlock(
                              hidden_size        = kwargs['d_model'],
                              num_attention_heads= kwargs['num_attention_heads'],
                              layer_idx          = kwargs['layer_idx'],
                              proj_groups        = kwargs.get('proj_groups',1),
                              attn_block_dtype   = kwargs.get('attn_block_dtype',torch.bfloat16),
                              mlp_dtype          = kwargs.get('mlp_dtype',torch.bfloat16),
                              mha_out_proj_bias  = kwargs.get('mha_out_proj_bias',True),
                              qkv_proj_bias      = kwargs.get('qkv_proj_bias',False),
                              use_flash_attn     = kwargs.get('use_flash_attn',False),  # flash attention requires CUDA 12.4
                              inner_size_multiple_of = 16,     # these five parameters belong to the MLP
                              mlp_activation         = 'gelu', # these five parameters belong to the MLP
                              evo2_style_activations = True,   # these five parameters belong to the MLP
                              model_parallel_size    = 1,      # these five parameters belong to the MLP
                              inner_mlp_size         = 256     # these five parameters belong to the MLP
            ).to(torch.bfloat16)
            
    def move(self):
        """Move every sub-module of the encoder to ``self.device``.

        Equivalent to calling ``self.to(self.device)`` but written out
        explicitly, presumably so that callers can move only parts of the
        encoder.

        WARNING
        -------
        This method is stale and will raise ``AttributeError``: it dereferences
        ``self.CenterMaxPool`` and ``self.header``, which are never assigned in
        ``__init__``.  It is not called anywhere in this repository.  Prefer
        ``self.to(device)``.
        """
        self.conv0.to(self.device)
        self.motif_embedding.to(self.device)
        # NOTE: the two lines below reference attributes that this class never
        # defines -- left in place because this task adds documentation only.
        self.CenterMaxPool.to(self.device)
        self.header.to(self.device)
        for layer in self.HyenaCluster:
            layer.to(self.device)
        
    def p_count(self):
        """Print the total and trainable parameter counts of the encoder.

        Returns
        -------
        None
            The counts are printed, not returned.  Useful mainly to confirm
            the d_model=128 distilled backbone against the d_model=256 task
            configuration (Section 2.1).
        """
        ttp=0
        tp=0
        for p in self.parameters():
            c=p.numel()
            if p.requires_grad == True:
                ttp+=c
            tp+=c
        print(f"Total trainable parameters: {ttp}")
        print(f"Total parameters: {tp}")
        return None
        
    def forward(self,x,return_emb=False):
        """Encode a batch of CRE windows into sequence-level embeddings.

        Parameters
        ----------
        x : torch.Tensor
            With ``filter_type='emb'``: integer token ids ``[B, L]``.
            Otherwise: one-hot DNA of shape ``[B, L, 4]``, with ``L = 512``
            for eScreen (a cCRE centre extended by 256 bp each way,
            Section 2.3.1).
        return_emb : bool, default False
            Not used -- the method always returns the pooled embedding.  It
            exists so that :meth:`eScreen_vX.forward` can forward its own flag
            without branching.

        Returns
        -------
        torch.Tensor
            Pooled embedding of shape ``[B, d_model]``, cast to ``float32``.

        Notes
        -----
        The motif-scan path is::

            [B, 512, 4]  --rearrange-->  [B, 4, 512]
                         --frozen conv0--> [B, 1274, 512]
                         --ReLU-->        [B, 1274, 512]
                         --rearrange-->   [B, 512, 1274]
                         --motif_embedding--> [B, 512, d_model]

        The ReLU discards negative motif evidence so that a channel reads as
        "this motif is present here" rather than as a signed deviation; this is
        what keeps the 1,274 channels interpretable as TF binding evidence
        (Section 2.3.6).  The four blocks are combined by plain residual
        addition, and the mean over the length axis yields the sequence-level
        embedding consumed by every downstream head (Section 2.1, "Output
        Representation").
        """
        
        if self.filter_type == 'emb':
            ### direct embedding: the shape changes from (b,l) to (b,l,d)
            x = self.emb_layer(x.to(torch.int))
        else:
            ### motif scan: the shape changes from (b,l,4) to (b,4,l) and then to (b, ck, l)
            x = rearrange( x, 'b l c -> b c l').to(torch.bfloat16)
            x = nn.functional.relu(self.conv0(x))
            x = rearrange( x, 'b c l -> b l c') # move channels to the last dim
            x = self.motif_embedding(x)         # map features to d_model
        
        ### through the stripedHyena and header layers the shape stays (batch,ls+lc,d_model)
        x = x.to(torch.bfloat16)
        for layer in self.HyenaCluster:
            x = x + layer(x)

        emb = x.mean(dim=1)
        return emb.float()
    

    
class FlexibleFFN(nn.Module):
    """Stack of Linear + GELU + Dropout blocks with a configurable depth.

    A small utility used to build prediction heads whose depth can be varied in
    ablations without rewriting the module.  Each entry of ``hidden_dims`` adds
    one ``Linear -> GELU -> Dropout`` unit; a final ``Linear`` maps to
    ``output_dim`` with no activation, so the output is a raw logit / regression
    value rather than a squashed one.

    Note that this class is not used by the production eScreen model -- the
    readout heads are written out explicitly in :class:`eScreen_vX`.

    Parameters
    ----------
    input_dim : int
        Input feature width.
    output_dim : int
        Output feature width.
    hidden_dims : list of int, optional
        Width of each hidden layer, in order.  Defaults to ``[output_dim]``,
        i.e. a single hidden layer of the output width.
    dropout : float, default 0.1
        Dropout probability applied after every hidden activation.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.1):
        """Assemble the layer stack.

        Parameters
        ----------
        input_dim : int
            Input feature width.
        output_dim : int
            Output feature width.
        hidden_dims : list of int, optional
            Hidden widths.  ``None`` means a single ``output_dim``-wide layer.
        dropout : float, default 0.1
            Dropout probability between layers.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [output_dim]
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """Apply the feed-forward stack.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[..., input_dim]``.

        Returns
        -------
        torch.Tensor
            Output of shape ``[..., output_dim]``, without a final activation.
        """
        return self.net(x)
    
    

class CellConditionedLoRA(nn.Module):
    """
    Cell-Conditioned LoRA: generate a low-rank transform per cell line

    Implements the cell-type conditioning described in Section 2.3.5 and
    Fig. 2d.  A frozen Geneformer embedding of the cell line is mapped to a
    pair of low-rank factors, which together define a per-sample linear map
    applied to the pooled sequence embedding::

        delta = x @ A @ B
          A = f_A(ct_emb),  A in [B, D, r]
          B = f_B(ct_emb),  B in [B, r, D]
        out = x + alpha * delta

    This keeps the encoder itself cell-agnostic while still letting the same
    CRE be scored differently in different cell lines -- the small rank ``r``
    (64 in the production model) limits the modulation to a few directions so
    it cannot override the sequence representation wholesale.

    Initialisation strategy: ``f_B`` ends in a zero-initialised linear layer, so
    ``B = 0`` and therefore ``delta = 0`` at step 0.  The module therefore starts
    out as an exact identity on the sequence embedding and only drifts away from
    it as training proceeds -- the same trick used in standard LoRA to avoid
    perturbing a pretrained representation at the start of fine-tuning.
    """
    def __init__(self, d_model, rank=4, alpha=1.0):
        """Build the two factor generators and the learnable scale.

        Parameters
        ----------
        d_model : int
            Width of the pooled sequence embedding (256 in production).
        rank : int, default 4
            LoRA rank ``r``.  The production model uses ``rank=64``.
        alpha : float, default 1.0
            Initial value of the learnable, scalar output scale.
        """
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        
        # produce the A matrix (down-projection)
        # f_A and f_B both squeeze the cell embedding through a 32-dimensional
        # bottleneck before fanning out to d_model * rank entries.  The
        # bottleneck is what keeps this conditioning cheap: the full per-sample
        # matrices would otherwise cost O(d_model^2) parameters, whereas here
        # the generator only has to learn a low-dimensional map from cell
        # identity to a subspace direction.
        self.f_A = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, d_model * rank)
        )
        
        # produce the B matrix (up-projection)
        self.f_B = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, rank * d_model)
        )
        
        # key init: zero the weights and bias of the last f_B layer
        # so initially B=0 and delta=0, leaving the representation unchanged
        nn.init.zeros_(self.f_B[-1].weight)
        nn.init.zeros_(self.f_B[-1].bias)
        
        # initialize A normally (small values)
        nn.init.normal_(self.f_A[-1].weight, std=0.02)
        nn.init.zeros_(self.f_A[-1].bias)
        
        # learnable scaling factor, initialized small
        self.alpha = nn.Parameter(torch.tensor(alpha))
    
    def forward(self, x, ct_emb):
        """Apply the cell-conditioned low-rank modulation.

        Parameters
        ----------
        x : torch.Tensor
            Sequence representation, either ``[B, L, D]`` (unpooled) or
            ``[B, D]`` (already pooled over the length axis).  Both are
            supported because :class:`eSCREEN_backbone` pools internally, but
            callers that hook the modulation before pooling can pass 3-D input.
        ct_emb : torch.Tensor
            Cell-type embedding of shape ``[B, D]``, already projected to the
            model width by the adapter in :class:`eScreen_vX`.

        Returns
        -------
        torch.Tensor
            ``x + alpha * delta`` with the same shape as the input ``x``.

        Notes
        -----
        ``torch.bmm`` is used rather than ``torch.einsum`` because the factors
        are per-sample (each row of the batch has its own ``A`` and ``B``),
        which matches batched matrix multiplication exactly and avoids
        materialising the full ``[B, D, D]`` transform.  The 2-D branch
        temporarily inserts a length axis of 1 so the same batched product can
        be reused, then squeezes it away.
        """
        B = x.size(0)
        
        # produce A and B
        A_flat = self.f_A(ct_emb)  # [B, D * rank]
        B_flat = self.f_B(ct_emb)  # [B, rank * D]
        
        A = A_flat.view(B, self.d_model, self.rank)  # [B, D, r]
        B = B_flat.view(B, self.rank, self.d_model)  # [B, r, D]
        
        # compute delta = x @ A @ B
        if x.dim() == 3:
            # [B, L, D] @ [B, D, r] @ [B, r, D] → [B, L, D]
            x_A = torch.bmm(x, A)  # [B, L, r]
            delta = torch.bmm(x_A, B)  # [B, L, D]
        else:
            # [B, D] @ [B, D, r] @ [B, r, D] → [B, D]
            x_A = torch.bmm(x.unsqueeze(1), A)  # [B, 1, r]
            delta = torch.bmm(x_A, B).squeeze(1)  # [B, D]
        
        return x + self.alpha * delta

    
class PrototypeDecoder(nn.Module):
    """Mixture-of-experts regression readout (Section 2.1, Fig. 2d).

    Instead of a single linear probe, the pooled embedding is scored by ``K``
    "prototype" heads and the results are combined by a softmax gate that is
    conditioned on the cell type.  Each prototype can therefore specialise in a
    particular regulatory regime (for example a cell-line-specific enhancer
    grammar) while the router decides how much of each regime to apply to the
    current cell line.

    Parameters
    ----------
    d_model : int
        Width of the pooled sequence embedding.
    K : int, default 8
        Number of prototype heads.  The production eScreen model uses ``K=64``
        (passed as ``MoE_K`` from :class:`eScreen_vX`).
    """

    def __init__(self, d_model, K=8):
        """Allocate the prototype weights and the cell-conditioned router.

        Parameters
        ----------
        d_model : int
            Input width.
        K : int, default 8
            Number of prototypes.
        """
        super().__init__()
        self.K = K
        # K prototype heads, each a d_model -> 1 linear layer (or small MLP)
        self.proto_W = nn.Parameter(torch.randn(K, d_model, 1) * 0.02)
        self.proto_b = nn.Parameter(torch.zeros(K, 1))
        # router: cell embedding -> attention scores
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, K)
        )

    def forward(self, h, ct_emb):
        """Score one embedding through the gated mixture of prototypes.

        Parameters
        ----------
        h : torch.Tensor
            Pooled sequence embedding, ``[B, d_model]``.
        ct_emb : torch.Tensor
            Cell-type embedding, ``[B, d_model]``.  Only the router consumes
            it; the prototype heads themselves are cell-agnostic.

        Returns
        -------
        torch.Tensor
            Predicted activity, ``[B, 1]``.

        Notes
        -----
        The gate is a softmax over the ``K`` prototypes, so the weights sum to
        one and the output is a convex combination of the individual prototype
        scores.  This makes the readout a genuine mixture rather than an
        ensemble, and it means no extra normalisation is needed on the logits.

        No sigmoid is applied even though the target lies in ``[0, 1]``: the
        regression target is already a smooth monotone transform of the RRA
        value (Section 2.3.2), and training minimises a smooth-L1 loss, which
        would be distorted by a saturating output activation.
        """
        # h: [B, d_model]
        attn_scores = self.router(ct_emb)          # [B, K]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, K]
        # per-prototype output: [B, K, 1]
        # batched: h [B,1,d_model] @ proto_W [K,d_model,1] -> [B,K,1]
        proto_out = torch.einsum('bd,kdo->bko', h, self.proto_W).squeeze(-1) + self.proto_b.squeeze(-1)  # [B, K]
        out = (proto_out * attn_weights).sum(dim=-1, keepdim=True)  # [B, 1]
        #return torch.sigmoid(out)
        return out
    

class eScreen_vX(nn.Module):
    """Full eScreen model: encoder + cell conditioning + readout heads.

    Assembles the components described in Section 2.1 and Fig. 2d into the
    trainable model, and carries the training / inference loops::

        DNA [B, 512, 4] --backbone--> [B, d_model]
        cell label [B]  --cell_emb--> [B, 256] --ct_adapter--> [B, d_model]
                                                       |
                                        CellConditionedLoRA
                                                       |
                                     +-----------------+-----------------+
                                     |                                   |
                             PrototypeDecoder (K=64)            cls_head (BCE)
                             -> activity in [0, 1]              -> logit

    The four ``freeze_*`` flags implement the training schedule of Section 2.2:
    the sequence encoder is distilled from HyenaDNA first and then held fixed
    while the task-specific heads are fitted, so the default task run uses
    ``freeze_backbone=True``.

    The auxiliary classification head is optional and only contributes to the
    loss when ``aux_bce_lambda > 0`` in :meth:`fit`.  The docstring in the
    source describes this as "identical to M2a version"; the name refers to an
    internal model-revision tag, not to anything in the manuscript.
    """

    def __init__(self, backbone, d_model, cell_emb, d_cell_emb=256,
                 cell_lora_rank=64, MoE_K=64,
                 freeze_backbone=False, freeze_cellemb=False,
                 freeze_celllora=False, freeze_header=False):
        """Wire the encoder, the cell-type pathway and the output heads.

        Parameters
        ----------
        backbone : eSCREEN_backbone
            The sequence encoder.  Normally already distilled (Section 2.2).
        d_model : int
            Model width, matching the backbone output.
        cell_emb : nn.Module or callable
            Geneformer-based cell-type embedder, mapping a cell-type index
            ``[B]`` to a vector of width ``d_cell_emb`` (Section 2.3.5).
        d_cell_emb : int, default 256
            Width of the raw Geneformer embedding.
        cell_lora_rank : int, default 64
            Rank of the cell-conditioned low-rank modulation.
        MoE_K : int, default 64
            Number of prototypes in the mixture-of-experts readout; 64 matches
            Section 2.1.
        freeze_backbone, freeze_cellemb, freeze_celllora, freeze_header : bool
            Which parameter groups to hold fixed.  ``freeze_cellemb`` controls
            both the raw embedder and the adapter that projects it to
            ``d_model``.  See :meth:`_apply_freezing`.
        """
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.freeze_backbone = freeze_backbone
        self.freeze_cellemb = freeze_cellemb
        self.freeze_celllora = freeze_celllora
        self.freeze_header = freeze_header

        self.cell_emb = cell_emb
        self.ct_adapter = nn.Sequential(
            nn.Linear(d_cell_emb, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.cell_lora = CellConditionedLoRA(d_model, rank=cell_lora_rank)
        self.output_MoE = PrototypeDecoder(d_model=d_model, K=MoE_K)
        self.output_header = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.LayerNorm(d_model * 4),
            nn.GELU(), nn.Linear(d_model * 4, 1),
        )
        self.cls_head = nn.Linear(d_model, 1)
        self._apply_freezing()

    def _apply_freezing(self):
        """Set ``requires_grad`` for every parameter group and fix BN state.

        Called once at the end of ``__init__``.  Two things happen here:

        1. ``requires_grad`` is toggled per group, so that the distilled
           encoder can stay frozen while the LoRA modulation and the heads are
           optimised (the schedule described in Sections 2.2 and 2.3).
        2. Any batch-norm layer inside the backbone has ``track_running_stats``
           disabled and is pinned to eval mode when the backbone is frozen.
           This matters because a frozen backbone must not silently keep
           updating its running statistics -- otherwise "frozen" would still
           change the function being computed.

        Notes
        -----
        ``freeze_cellemb`` governs both the raw Geneformer embedder and the
        ``ct_adapter`` that projects it to the model width; the two are treated
        as a single unit because the adapter is meaningless without the
        embedding it consumes.
        """
        for p in self.backbone.parameters():
            p.requires_grad = not self.freeze_backbone
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.track_running_stats = not self.freeze_backbone
                if self.freeze_backbone:
                    m.eval()
                else:
                    m.train()
        for p in self.cell_emb.parameters():
            p.requires_grad = not self.freeze_cellemb
        for p in self.ct_adapter.parameters():
            p.requires_grad = not self.freeze_cellemb
        for p in self.cell_lora.parameters():
            p.requires_grad = not self.freeze_celllora
        for p in self.output_MoE.parameters():
            p.requires_grad = not self.freeze_header
        for p in self.output_header.parameters():
            p.requires_grad = not self.freeze_header
        for p in self.cls_head.parameters():
            p.requires_grad = not self.freeze_header

    def p_count(self):
        """Report parameter counts overall and per top-level sub-module.

        Returns
        -------
        int
            Total number of trainable parameters.

        Notes
        -----
        The per-module breakdown is grouped by the first component of each
        parameter name, so it reports one line per top-level attribute
        (``backbone``, ``ct_adapter``, ``cell_lora``, ``output_MoE``,
        ``output_header``, ``cls_head``).  This is the quickest way to confirm
        that freezing actually took effect.
        """
        tp = sum(p.numel() for p in self.parameters())
        ttp = sum(p.numel() for p in self.parameters() if p.requires_grad)
        module_ttps = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                module = name.split('.')[0]
                module_ttps[module] = module_ttps.get(module, 0) + p.numel()
        print(f"Total params: {tp:,}, Trainable: {ttp:,}")
        print(f"Trainable by module: {module_ttps}")
        return ttp

    def forward(self, x, ct, return_cls=False, return_emb=False):
        """Run the full model on a batch of CREs and cell types.

        Parameters
        ----------
        x : torch.Tensor
            One-hot DNA, ``[B, 512, 4]``.  Cast to ``float`` by the training
            loop; the backbone internally downcasts to ``bfloat16``.
        ct : torch.Tensor
            Cell-type indices, ``[B]``, indexing the Geneformer embedder.
        return_cls : bool, default False
            If true, also return the auxiliary classification logit.  Used to
            supply the optional BCE branch during training.
        return_emb : bool, default False
            If true, also return the pooled embedding.  Ignored when
            ``return_cls`` is true, because the classification path takes
            precedence in the branch below.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            ``reg`` of shape ``[B]``, or ``(reg, cls)`` / ``(reg, emb)``.

        Notes
        -----
        The cell embedding is recomputed for every batch rather than cached:
        it is a pure function of the integer cell-type index and is cheap
        relative to the backbone, so caching would only complicate the
        transfer between devices.

        ``emb.dim() == 3`` is a defensive branch.  :class:`eSCREEN_backbone`
        already mean-pools along the length axis, so in the standard path the
        embedding is 2-D and this branch is skipped; it exists because
        :class:`CellConditionedLoRA` preserves the rank of its input, so a
        caller that swaps in an unpooled encoder still gets a pooled result
        here.
        """
        emb = self.backbone(x)
        ct_emb_0 = self.ct_adapter(self.cell_emb(ct).to(x.device))
        emb = self.cell_lora(emb, ct_emb_0)
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        reg = self.output_MoE(emb, ct_emb_0).flatten()
        if return_cls:
            cls = self.cls_head(emb).flatten()
            return reg, cls
        if return_emb:
            return reg, emb
        return reg

    def fit(
        self,
        train_data,
        val_data=None,
        batch_size=256,
        epochs=200,
        optimizer=None,
        scheduler=None,
        check_step=500,
        earlystop=20,
        use_boost=True,
        t=0.6,
        metrics=None,
        task='reg',
        device='cpu',
        save_name='./torch_logs/model',
        aux_bce_lambda=0.0,
        aux_loss_type='bce',
        focal_gamma=2.0,
        focal_alpha=None,
        pos_weight=None,
        label_smoothing=0.0,
        max_grad_norm=1.0,
        sampler='boost',
        val_score_fn=None,
    ):
        """Train the model with early stopping and periodic checkpointing.

        The optimisation setup follows Section 2.3: a smooth-L1 regression loss
        on the transformed RRA target, optionally augmented by a binary
        cross-entropy term, with class-balanced oversampling handled by
        :class:`utils.boost_sampler` (Section 2.3.4).

        Parameters
        ----------
        train_data, val_data : dict
            In-memory dataset.  Required keys are ``'sequence'`` (one-hot
            ``[N, 512, 4]``), ``'y'`` (regression target ``[N]``), ``'label'``
            (binary label, needed for the BCE branch) and ``'cell_type'``
            (``[N]`` integer) plus the scalar ``'length'``.  An optional
            ``'_boost_labels'`` key overrides ``'label'`` as the grouping
            variable for the sampler.
        batch_size : int, default 256
            Mini-batch size.
        epochs : int, default 200
            Maximum number of epochs.
        optimizer, scheduler : optional
            Torch objects.  Defaults to ``AdamW(lr=3e-4, weight_decay=0.01)``.
            The scheduler, if given, is stepped once per mini-batch -- not per
            epoch -- so a per-epoch schedule must be configured accordingly.
        check_step : int, default 500
            Validation is triggered every ``check_step`` optimisation steps
            rather than at epoch boundaries, which keeps the check rate stable
            when the dataset size changes.
        earlystop : int, default 20
            Number of consecutive validation checks without improvement before
            training stops and the best checkpoint is restored.
        use_boost : bool, default True
            **Ignored.**  It is immediately shadowed by a local reset inside
            this method, so the sampler is selected solely by ``sampler``.
            Kept in the signature only for backward compatibility.
        t : float, default 0.6
            Temperature of the class-balanced sampler; the paper uses 0.45
            (Section 2.3.4).
        metrics : callable, optional
            ``metrics(preds, y_true) -> float``, higher is better.  It is
            required for checkpointing to work: when it is ``None`` the
            validation score is NaN, every comparison against the running best
            is therefore false, and **no ``.best.pt`` file is written**.
        task : {'reg', 'cls'}
            Which loss to optimise.  ``'reg'`` is the eScreen task.
        device : str, default 'cpu'
            Torch device string.
        save_name : str, default './torch_logs/model'
            Path prefix.  Writes ``.best.pt``, one ``.<epoch>.pt`` per epoch,
            and ``.final.pt`` at the end.
        aux_bce_lambda : float, default 0.0
            Weight of the auxiliary classification loss.  Zero disables it;
            a positive value requires ``train_data['label']``.
        aux_loss_type : {'bce', 'focal'}
            Form of the auxiliary term.
        focal_gamma, focal_alpha : float
            Focal-loss parameters, used only when ``aux_loss_type='focal'``.
        pos_weight : float, optional
            Positive-class weight for the BCE terms, for class imbalance.  Only
            applied when greater than zero.
        label_smoothing : float, default 0.0
            Clamped to ``[0, 0.49]`` and applied to the classification targets
            only; regression targets are never smoothed.
        max_grad_norm : float, optional
            Gradient-norm clip.  Values that are ``None`` or non-positive
            disable clipping.
        sampler : {'boost', 'label_boost', 'label', 'none'}
            Sampling strategy.  ``'boost'`` (default) uses the class-balanced
            sampler; ``'none'`` falls back to shuffled sampling with wrap-around
            padding of the last epoch.

        Returns
        -------
        None
            The model is updated in place; checkpoints are written to disk.

        Notes
        -----
        ``sample_num`` rounds the epoch up to a whole number of batches and the
        sampler draws that many indices, but the training loop only iterates
        over ``train_data['length']`` examples, so the trailing surplus indices
        are simply unused.  The effect is a slight bias in the last partial
        batch of each epoch.
        """
        if optimizer is None:
            optimizer = AdamW(self.parameters(), lr=3e-4, weight_decay=0.01)

        best_valid_acc = float('-inf')
        count = 0
        valid_count = 0

        # The argument ``use_boost`` is deliberately shadowed here: the local
        # flag is reset to False and then only re-enabled from ``sampler``.
        # Callers must therefore switch sampling off with sampler='none', not
        # with use_boost=False.
        use_boost = False
        sampler_mode = str(sampler).lower().strip()
        if sampler_mode == 'none':
            pass
        elif sampler_mode in ('label_boost', 'label', 'boost') or use_boost:
            use_boost = True
        else:
            raise ValueError(f"Unknown sampler: {sampler!r}")

        def _make_booster():
            """Build the class-balanced sampler, or None when sampling is off.

            Chooses the grouping variable for the sampler in priority order:

            1. ``train_data['_boost_labels']`` if present -- an explicit override,
               typically the binary functional/non-functional label;
            2. ``train_data['label']`` when ``sampler`` is one of the
               ``'label*'`` modes;
            3. ``train_data['weight']`` if present -- lets callers up-weight by a
               continuous score instead of a class;
            4. ``train_data['label']`` as the fallback.

            Returns
            -------
            utils.boost_sampler or None
                ``None`` when ``use_boost`` is false, in which case the training
                loop falls back to shuffled sampling with wrap-around padding.
            """
            if not use_boost:
                return None
            if '_boost_labels' in train_data:
                labels = train_data['_boost_labels']
            elif sampler_mode in ('label_boost', 'label'):
                labels = train_data['label']
            elif 'weight' in train_data:
                labels = train_data['weight']
            else:
                labels = train_data['label']
            return boost_sampler(labels, smoothing=1.0, temperature=t, seed=114514)

        booster = _make_booster()

        pw_tensor = None
        if pos_weight is not None and float(pos_weight) > 0:
            pw_tensor = torch.tensor(float(pos_weight), device=device)

        label_smooth = max(0.0, min(float(label_smoothing), 0.49))
        grad_clip = float(max_grad_norm) if max_grad_norm is not None and float(max_grad_norm) > 0 else None

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_step = 0
            sample_num = (train_data['length'] // batch_size + 1) * batch_size

            if use_boost and booster is not None:
                idx = booster.get(sample_num)
            else:
                pad = sample_num - train_data['length']
                idx = np.random.permutation(train_data['length'])
                idx = np.concatenate([idx, np.random.choice(idx, size=pad, replace=True)])

            for i in range(0, train_data['length'], batch_size):
                batch_idx = idx[i:i + batch_size]
                x = torch.tensor(train_data['sequence'][batch_idx], dtype=torch.float, device=device)
                y = torch.tensor(train_data['y'][batch_idx], dtype=torch.float, device=device).flatten()
                ct = torch.tensor(train_data['cell_type'][batch_idx], dtype=torch.int, device=device)

                self.train()
                optimizer.zero_grad()

                p_reg, p_cls = self(x, ct, return_cls=True)

                if task == 'cls':
                    y_cls = y
                    if label_smooth > 0:
                        y_cls = y * (1.0 - label_smooth) + 0.5 * label_smooth
                    loss = F.binary_cross_entropy_with_logits(
                        p_cls, y_cls, pos_weight=pw_tensor, reduction='mean'
                    )
                elif task == 'reg':
                    loss = F.smooth_l1_loss(p_reg, y, beta=0.1)
                    if aux_bce_lambda > 0:
                        if 'label' not in train_data:
                            raise KeyError("aux_bce_lambda > 0 requires train_data['label']")
                        lab = torch.tensor(
                            train_data['label'][batch_idx], dtype=torch.float, device=device
                        )
                        if label_smooth > 0:
                            lab = lab * (1.0 - label_smooth) + 0.5 * label_smooth
                        if str(aux_loss_type).lower() in ('focal', 'focal_bce'):
                            aux = self._focal_bce_with_logits(
                                p_cls, lab, gamma=float(focal_gamma),
                                alpha=focal_alpha, pos_weight=pw_tensor,
                            )
                        else:
                            aux = F.binary_cross_entropy_with_logits(
                                p_cls, lab, pos_weight=pw_tensor, reduction='mean'
                            )
                        loss = loss + aux_bce_lambda * aux
                else:
                    raise ValueError(f"Unknown task: {task!r}")

                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                epoch_loss += loss.item()
                epoch_step += 1
                avg_loss = epoch_loss / epoch_step
                print(
                    f"Step/Epoch [{i+1}/{epoch+1}], Loss: {avg_loss:.4f}, Step: {loss.item():.4f} ",
                    end='\r',
                )

                valid_count += 1
                if valid_count >= check_step and val_data is not None:
                    if val_score_fn is not None:
                        valid_acc = float(val_score_fn(self, val_data, device))
                    else:
                        preds, y_reg = self.predict(
                            val_data, batch_size=256, device=device, verbose=True, with_true=True
                        )
                        valid_acc = metrics(preds, y_reg) if metrics is not None else float('nan')

                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        count = 0
                        torch.save(self.state_dict(), save_name + '.best.pt')
                        print(f"Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Val: {valid_acc:.4f}  ↑")
                    else:
                        count += 1
                        print(f"Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Val: {valid_acc:.4f}  -")
                        if count > earlystop:
                            print(f'  Early stop at epoch {epoch+1}, best val: {best_valid_acc:.4f}')
                            self.load_state_dict(
                                torch.load(save_name + '.best.pt', map_location=device, weights_only=False)
                            )
                            return
                    valid_count = 0

            torch.save(self.state_dict(), f'{save_name}.{epoch}.pt')
        torch.save(self.state_dict(), save_name + '.final.pt')

    @staticmethod
    def _focal_bce_with_logits(pred, target, gamma=2.0, alpha=None, pos_weight=None, reduction='mean'):
        """Binary focal loss computed directly from logits.

        Focal loss down-weights examples that are already classified
        confidently, which keeps the large number of easy negative CREs from
        dominating the auxiliary objective.

        Parameters
        ----------
        pred : torch.Tensor
            Raw logits, any shape.
        target : torch.Tensor
            Binary targets in ``[0, 1]``, same shape as ``pred``.
        gamma : float, default 2.0
            Focusing exponent.  ``gamma=0`` recovers plain BCE; larger values
            suppress well-classified examples more aggressively.
        alpha : float, optional
            Class-prior balancing factor in ``[0, 1]``, applied as
            ``alpha`` on positives and ``1 - alpha`` on negatives.
        pos_weight : torch.Tensor, optional
            Passed through to the underlying ``binary_cross_entropy_with_logits``.
        reduction : {'mean', 'sum', 'none'}
            How to aggregate the element-wise loss.

        Returns
        -------
        torch.Tensor
            Scalar loss for ``'mean'``/``'sum'``, or the element-wise loss for
            ``'none'``.

        Notes
        -----
        ``pt = exp(-BCE)`` recovers the predicted probability of the true class
        from the unreduced BCE value, which avoids a second sigmoid pass and
        keeps the computation numerically identical to the standard focal-loss
        formulation.
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight, reduction='none')
        pt = torch.exp(-bce)
        focal = (1.0 - pt) ** gamma * bce
        if alpha is not None:
            alpha_t = target * alpha + (1 - target) * (1 - alpha)
            focal = alpha_t * focal
        if reduction == 'mean':
            return focal.mean()
        elif reduction == 'sum':
            return focal.sum()
        return focal

    def get_emb(self, data, batch_size=16, device='cpu', verbose=True):
        """Extract pooled embeddings for a whole dataset.

        Runs a forward pass under ``torch.no_grad()`` and collects the
        cell-conditioned embedding that feeds the readout heads -- i.e. the
        representation *after* the LoRA modulation, not the raw backbone
        output.

        Parameters
        ----------
        data : dict
            Must contain ``'sequence'``, ``'cell_type'`` and ``'length'``.
        batch_size : int, default 16
            Inference batch size.  Smaller than the training default because
            this path is only used for visualisation and downstream analysis.
        device : str, default 'cpu'
            Torch device string.
        verbose : bool, default True
            Show a ``tqdm`` progress bar.

        Returns
        -------
        numpy.ndarray
            Array of shape ``[N, d_model]``.

        Notes
        -----
        The model is switched to ``eval()`` mode and is **not** restored
        afterwards, so a caller that later resumes training must call
        ``model.train()`` explicitly.
        """
        embs = []
        self.eval()
        indices = range(0, data['length'], batch_size)
        pbar = tqdm(indices) if verbose else indices
        with torch.no_grad():
            for i in pbar:
                x = torch.tensor(data['sequence'][i:i + batch_size], dtype=torch.float, device=device)
                ct = torch.tensor(data['cell_type'][i:i + batch_size], dtype=torch.int, device=device)
                _,emb = self(x, ct, return_emb=True)
                embs.extend(emb.cpu().numpy())
        return np.stack(embs)
    
    def predict(self, data, batch_size=16, device='cpu', verbose=True, with_true=False):
        """Predict activity scores for a whole dataset.

        Parameters
        ----------
        data : dict
            Must contain ``'sequence'``, ``'cell_type'`` and ``'length'``;
            ``'y'`` is needed only when ``with_true`` is true.
        batch_size : int, default 16
            Inference batch size.
        device : str, default 'cpu'
            Torch device string.
        verbose : bool, default True
            Show a ``tqdm`` progress bar.
        with_true : bool, default False
            Also return the stored targets, for direct metric computation.

        Returns
        -------
        y_pred : numpy.ndarray
            Predicted activity, shape ``[N]``.
        y_true : numpy.ndarray
            Targets, shape ``[N]``; empty when ``with_true`` is false.

        Notes
        -----
        The classification head is explicitly bypassed (``return_cls=False``),
        so the returned values are the continuous MoE regression output and
        can be compared against the transformed RRA target directly.  As with
        :meth:`get_emb`, the model is left in ``eval()`` mode.
        """
        y_pred, y_true = [], []
        self.eval()
        indices = range(0, data['length'], batch_size)
        pbar = tqdm(indices) if verbose else indices
        with torch.no_grad():
            for i in pbar:
                x = torch.tensor(data['sequence'][i:i + batch_size], dtype=torch.float, device=device)
                ct = torch.tensor(data['cell_type'][i:i + batch_size], dtype=torch.int, device=device)
                preds = self(x, ct, return_cls=False)
                y_pred.extend(preds.cpu().numpy())
                if with_true:
                    y_true.extend(data['y'][i:i + batch_size])
        return np.array(y_pred), np.array(y_true)    


    def attribute(self,x,ct,batch_size=1,input_dtype=torch.float,output_file='./temp.npdat',device='cpu', verbose=True):
        """Compute nucleotide-level Input x Gradient attributions.

        Implements the attribution step of Supplementary Methods Section 3.2.
        For every CRE the method integrates the gradient of the model output
        along the straight path from an all-zero baseline to the one-hot input,
        then multiplies the resulting attribution by the input itself
        ("Input x Gradient"), so that only the nucleotide actually present at
        each position can carry a non-zero contribution.

        Parameters
        ----------
        x : array-like
            One-hot DNA, ``[N, 512, 4]``.
        ct : array-like
            Cell-type indices, ``[N]``.  Fed to the model through
            ``additional_forward_args`` so that the attribution is computed
            under the matching cell-type condition.
        batch_size : int, default 1
            Sequences per attribution call.  The default is deliberately 1:
            Integrated Gradients keeps an internal batch dimension for the
            ``n_steps`` interpolation points, so memory grows as
            ``batch_size * n_steps * 512 * 4``.
        input_dtype : torch.dtype, default torch.float
            Dtype of the input tensor handed to the model.
        output_file : str, default './temp.npdat'
            Destination of the result.  The name suggests ``.npy`` but the data
            is written as a **raw float32 memmap of shape ``[N, 512]``**, i.e. a
            flat binary blob with no header -- read it back with
            ``np.memmap(output_file, dtype='float32', mode='r', shape=(N, 512))``,
            not with ``np.load``.
        device : str, default 'cpu'
            Torch device string.
        verbose : bool, default True
            Show a ``tqdm`` progress bar.

        Returns
        -------
        None
            Results are streamed to ``output_file``; nothing is returned.  The
            memmap is flushed every 100 batches and once more at the end, so a
            partially computed file remains readable if the run is interrupted.

        Notes
        -----
        The integration path uses the **all-zero sequence** as baseline, which
        corresponds to "no nucleotide present at any position" rather than to a
        shuffled or dinucleotide-matched background.  Section 3.2 justifies this
        choice: it requires no extra background dataset and allows attributions
        to be computed independently for a single sequence.

        ``n_steps=30`` with ``method='gausslegendre'`` matches the protocol
        reported in Section 3.2.  ``return_convergence_delta=True`` asks captum
        for the completeness-error estimate as well, but this method discards
        it (the underscore in ``batch_attr, _ = ...``); callers who need the
        convergence check must re-implement the loop.

        The attribution is reduced over the four nucleotide channels with
        ``sum(axis=-1)``, which yields one signed scalar per base position.  A
        positive value means the nucleotide pushes predicted activity up
        relative to the baseline.
        """
        self.eval()
        SAMPLENUM = len(x)
        weight = np.memmap(output_file, dtype='float32', mode='w+', shape=(SAMPLENUM, 512))
        ig = IntegratedGradients(self.forward)

        indices = range(0, len(x), batch_size)
        pbar = tqdm(indices) if verbose else indices

        for i in pbar:
            torch.cuda.empty_cache()
            batch_x  = torch.tensor(x[i:i+batch_size],dtype=input_dtype).to(device).requires_grad_()
            batch_ct = torch.tensor(ct[i:i+batch_size],dtype=torch.int).to(device).detach()
            baselines = torch.zeros_like(batch_x)

            batch_attr, _ = ig.attribute(
                batch_x,
                baselines=baselines,
                additional_forward_args=batch_ct,
                return_convergence_delta=True,
                method="gausslegendre",
                n_steps=30
            )
            weight[i:i+batch_size] = (batch_attr.detach().cpu().numpy()*batch_x.detach().cpu().numpy()).sum(axis=-1)
            if i%100==0:weight.flush()
        weight.flush()
