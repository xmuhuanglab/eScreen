"""Cell-type embeddings for eScreen.

eScreen conditions its prediction on *which cell line* a CRE is being scored
in.  That conditioning vector is produced here, and the module contains **two
independent implementations** of it, only one of which is used in the paper:

1. :class:`CellEncoder` / :class:`ExprDecoder` / :class:`CellEmb` -- a small
   transformer trained from scratch with a masked-expression-modelling
   objective on bulk/single-cell expression vectors.  Self-contained and
   dependency-light, but *not* the path used for the reported results.

2. :class:`Geneformer` -- a thin wrapper around the pretrained Geneformer V1
   model.  **This is the production path** described in Supplementary Methods
   Section 2.3.5: DepMap bulk RNA-seq profiles are tokenised the way Geneformer
   prescribes, the hidden states of the penultimate layer are mean-pooled over
   the valid gene tokens, and the resulting vector is used as the cell-type
   embedding (normalisation target sum 10,000).

The reason for preferring Geneformer is stated in Section 2.3.5: it was
pretrained on roughly 30 million single-cell transcriptomes and therefore
already encodes the hierarchical structure of gene regulatory programmes, which
a small supervised encoder trained on a few hundred cell lines cannot recover.

Downstream, the embedding is projected to the model width and fed to
``CellConditionedLoRA`` as well as to the prototype router of the MoE readout
(see ``escreen.py``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class CellEncoder(nn.Module):
    """Expression-to-embedding transformer encoder (alternative to Geneformer).

    Not used in the published pipeline.  Kept because it provides the same
    interface as the Geneformer wrapper -- ``(gene_ids, exprs) -> [B, d_model]``
    -- which makes it a drop-in substitute for quick experiments when the
    Geneformer checkpoints are not available.

    Parameters
    ----------
    num_genes : int
        Size of the gene vocabulary.  The default used by :class:`CellEmb`
        (19,215) is the number of protein-coding genes in the Geneformer
        dictionary.
    d_model : int, default 256
        Embedding width.
    nhead : int, default 4
        Number of attention heads.
    num_layers : int, default 4
        Number of transformer-encoder layers.
    """

    def __init__(self, num_genes, d_model=256, nhead=4, num_layers=4):
        """Build the gene embedding, the expression MLP and the encoder stack.

        Parameters
        ----------
        num_genes : int
            Gene vocabulary size.
        d_model : int, default 256
            Embedding width.
        nhead : int, default 4
            Attention heads.
        num_layers : int, default 4
            Encoder depth.
        """
        super().__init__()

        # gene identity
        self.gene_emb = nn.Embedding(num_genes, d_model)

        # expression encoder
        self.expr_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # transformer encoder (no positional encoding!)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # attention pooling
        self.attn_pool = nn.Linear(d_model, 1)

    def forward(self, gene_ids, exprs, log1p = True):
        """Encode a batch of expression vectors into cell embeddings.

        Parameters
        ----------
        gene_ids : torch.Tensor
            Gene indices, ``[B, N]``, ``int64``.  ``N`` is the number of genes
            retained per cell after normalisation and truncation.
        exprs : torch.Tensor
            Matching expression values, ``[B, N]``, float.
        log1p : bool, default True
            Whether ``exprs`` has already been log1p-transformed.  If False the
            transform is applied here.

        Returns
        -------
        cell_emb : torch.Tensor
            Attention-pooled cell embedding, ``[B, d_model]``.
        x : torch.Tensor
            Per-gene token embeddings, ``[B, N, d_model]``.  Returned because
            the expression decoder needs them for the masked-modelling loss.

        Notes
        -----
        There is deliberately **no positional encoding**.  The gene axis is a
        set, not a sequence: genes are sorted by expression level, so their
        position carries no biological meaning and a positional code would
        inject a spurious signal (a gene's "position" would change whenever the
        ranking changed).

        Expression enters multiplicatively as a gate, ``g_emb * sigmoid(e_emb)``,
        rather than additively.  The sigmoid bounds the gate to ``(0, 1)``, so a
        gene can be strongly attenuated when it is not expressed but can never
        be sign-flipped, which keeps the gene-identity direction intact.
        """

        # gene embedding
        g_emb = self.gene_emb(gene_ids)  # (B, N, d)

        # expression embedding
        if not log1p: # if the input is not log1p
            exprs = torch.log1p(exprs)  # this is crucial!
        e_emb = self.expr_mlp(exprs.unsqueeze(-1))  # (B, N, d)

        # gating (stronger than plain addition)
        x = g_emb * torch.sigmoid(e_emb)

        # transformer
        x = self.transformer(x)

        # attention pooling
        attn = torch.softmax(self.attn_pool(x), dim=1)  # (B, N, 1)
        cell_emb = (attn * x).sum(dim=1)  # (B, d)

        return cell_emb, x  # x is used for the mask task
    
def mask_expression(exprs, mask_ratio=0.15):
    """Randomly zero a fraction of expression values for masked training.

    Simulates dropout of individual gene measurements so that the decoder has
    to reconstruct them from the remaining context -- the masked-modelling
    pretext used by the self-supervised variant in this module.

    Parameters
    ----------
    exprs : torch.Tensor
        Expression matrix, ``[B, N]``.
    mask_ratio : float, default 0.15
        Fraction of entries to hide.

    Returns
    -------
    masked_exprs : torch.Tensor
        Copy of ``exprs`` with the selected entries set to zero.  The input
        tensor is not modified.
    mask : torch.Tensor
        Boolean tensor, ``[B, N]``, True where a value was hidden.  Used to
        restrict the reconstruction loss to the masked positions.

    Notes
    -----
    Masking by setting values to exactly zero keeps the gene present in the
    input as an all-zero token; the model therefore still knows *which* gene was
    unobserved, it only lacks the magnitude.  This is intentional -- it mirrors
    Geneformer's masked-learning setup, where the gene token remains in the
    sequence.
    """
    mask = torch.rand(exprs.shape, device=exprs.device) < mask_ratio
    masked_exprs = exprs.clone()
    masked_exprs[mask] = 0.0
    return masked_exprs, mask

class ExprDecoder(nn.Module):
    """Reconstruct per-gene expression from transformer token embeddings.

    The decoding half of the masked-modelling pretext: each gene token is
    mapped back to a scalar expression value, and the error on the masked
    positions provides the training signal for :class:`CellEncoder`.

    Parameters
    ----------
    d_model : int
        Width of the token embeddings produced by :class:`CellEncoder`.
    """

    def __init__(self, d_model):
        """Build the two-layer scalar regression head.

        Parameters
        ----------
        d_model : int
            Token-embedding width.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        """Decode token embeddings into expression values.

        Parameters
        ----------
        x : torch.Tensor
            Token embeddings, ``[B, N, d_model]``.

        Returns
        -------
        torch.Tensor
            Predicted expression, ``[B, N]``.  The trailing singleton axis is
            squeezed away, so callers get a matrix aligned with ``exprs``.
        """
        return self.mlp(x).squeeze(-1)  # (B, N)
    
def augment(exprs):
    """Apply gene dropout plus additive Gaussian noise to expression values.

    A light data augmentation for the self-supervised encoder: zeroing 10% of
    the genes mimics panel dropout, while the noise term prevents the model from
    keying on exact quantised values of the bulk RNA-seq input.

    Parameters
    ----------
    exprs : torch.Tensor
        Expression matrix, ``[B, N]``.

    Returns
    -------
    torch.Tensor
        Augmented copy of the input.  The original tensor is not modified.

    Notes
    -----
    The noise is scaled by 0.1 in the *transformed* space.  Since the input is
    expected to be log1p-transformed, a unit of noise here corresponds to a
    multiplicative change of about ``exp(0.1)`` in raw counts, i.e. roughly 10%.
    """
    # dropout genes
    drop_mask = torch.rand(exprs.shape, device=exprs.device) < 0.1
    exprs = exprs.clone()
    exprs[drop_mask] = 0

    # add noise
    noise = torch.randn_like(exprs) * 0.1
    return exprs + noise


class CellEmb(nn.Module):
    """Masked-expression autoencoder producing cell embeddings.

    Bundles :class:`CellEncoder` and :class:`ExprDecoder` so that a single call
    returns both the cell embedding (needed downstream by eScreen) and the
    reconstruction (needed for the training objective).

    Not used in the published pipeline -- see the module docstring.  It exists
    as a self-contained alternative to the Geneformer wrapper below.

    Parameters
    ----------
    num_genes : int, default 19215
        Gene vocabulary size, matching the Geneformer gene dictionary.
    d_model : int, default 256
        Embedding width.
    """

    def __init__(self, num_genes=19215, d_model=256):
        """Wire the encoder and the expression decoder.

        Parameters
        ----------
        num_genes : int, default 19215
            Gene vocabulary size.
        d_model : int, default 256
            Embedding width.
        """
        super().__init__()
        self.encoder = CellEncoder(num_genes=num_genes)
        self.decoder = ExprDecoder(d_model=d_model)
    
    def forward(self, gene_ids, exprs, mask_ratio=0.15):
        """Run the masked-modelling forward pass.

        Parameters
        ----------
        gene_ids : torch.Tensor
            Gene indices, ``[B, N]``.
        exprs : torch.Tensor
            Expression values, ``[B, N]``.
        mask_ratio : float, default 0.15
            Fraction of entries hidden before encoding.

        Returns
        -------
        pred_expr : torch.Tensor
            Reconstructed expression, ``[B, N]``.
        cell_emb : torch.Tensor
            Cell embedding, ``[B, d_model]``.
        mask : torch.Tensor
            Boolean mask, ``[B, N]``, marking the hidden entries.  The training
            loss should be evaluated **only** on ``mask`` -- the unmasked
            positions are trivially reconstructible as they are copied straight
            through the encoder input, so including them would dominate the
            gradient with a near-identity task.
        """
        # Masked modeling forward
        masked_exprs, mask = mask_expression(exprs, mask_ratio)
        cell_emb, token_emb = self.encoder(gene_ids, masked_exprs)
        pred_expr = self.decoder(token_emb)
        
        return pred_expr, cell_emb, mask
    
    def encode_cell(self, gene_ids, exprs):
        """Encode only, no decoding; used to obtain cell representations.

        Skips the masking step, so the encoder sees the full expression vector.
        This is the method to call when the embedding is needed for inference
        rather than for training.

        Parameters
        ----------
        gene_ids : torch.Tensor
            Gene indices, ``[B, N]``.
        exprs : torch.Tensor
            Expression values, ``[B, N]``, unmasked.

        Returns
        -------
        torch.Tensor
            Cell embedding, ``[B, d_model]``.
        """
        cell_emb, _ = self.encoder(gene_ids, exprs)
        return cell_emb
    
    def decode(self, token_emb):
        """Decode only, given already-computed token embeddings.

        Decoupled from :meth:`encode_cell` so that the decoder can be trained or
        probed separately without re-running the encoder.

        Parameters
        ----------
        token_emb : torch.Tensor
            Per-gene token embeddings, ``[B, N, d_model]``.

        Returns
        -------
        torch.Tensor
            Reconstructed expression, ``[B, N]``.
        """
        return self.decoder(token_emb)
    

    
    
##################################################################
## Geneformer can also be used to generate embeddings
##################################################################
import pickle
import pandas as pd
import numpy as np
from transformers import BertForMaskedLM

class Geneformer(nn.Module):
    
    def __init__(self, TOKEN_DICTIONARY_FILE, GENE_MEDIAN_FILE, ENSEMBL_DICTIONARY_FILE, 
                 ENSEMBL_MAPPING_FILE, MODEL_PATH, PREDIFINED_EMB=None, model_version="V1",device="cpu"):

        # load the model
        model = BertForMaskedLM.from_pretrained(
            MODEL_PATH, 
            output_hidden_states=True
        )        
        
        super().__init__()
        
        self.TOKEN_DICTIONARY_FILE = TOKEN_DICTIONARY_FILE
        self.GENE_MEDIAN_FILE = GENE_MEDIAN_FILE
        self.ENSEMBL_DICTIONARY_FILE = ENSEMBL_DICTIONARY_FILE
        self.ENSEMBL_MAPPING_FILE = ENSEMBL_MAPPING_FILE
        self.MODEL_PATH = MODEL_PATH
        self.model_version = model_version
        self.model = model
        
        # V1 vs V2 configuration
        if self.model_version == "V1":
            self.MAX_LENGTH = 2048
            self.SPECIAL_TOKEN = False
        elif self.model_version == "V2":
            self.MAX_LENGTH = 4096
            self.SPECIAL_TOKEN = True
        else:
            raise ValueError(f"model_version must be 'V1' or 'V2', got {model_version}")
        
        # load the gene -> token dictionary
        with open(self.TOKEN_DICTIONARY_FILE, 'rb') as f:
            self.gene_token_dict = pickle.load(f)
        self.token_gene_dict = {v: k for k, v in self.gene_token_dict.items()}
        
        # get special token IDs (may not exist in V1)
        self.cls_token_id = self.gene_token_dict.get('<cls>')
        self.eos_token_id = self.gene_token_dict.get('<eos>')
        self.pad_token_id = self.gene_token_dict.get('<pad>', 0)
        
        # load gene medians (used for normalization)
        if self.GENE_MEDIAN_FILE:
            with open(self.GENE_MEDIAN_FILE, 'rb') as f:
                self.gene_median_dict = pickle.load(f)
        
        # load the dict mapping gene symbols to Ensembl IDs        
        if self.ENSEMBL_DICTIONARY_FILE:
            with open(self.ENSEMBL_DICTIONARY_FILE, 'rb') as f:
                self.gene_symbol_to_ensembl = pickle.load(f)    
        
        # load the Ensembl ID mapping dict (for version stripping and ID normalization)
        if self.ENSEMBL_MAPPING_FILE:
            with open(self.ENSEMBL_MAPPING_FILE, 'rb') as f:
                self.ensembl_mapping_dict = pickle.load(f)
        else:
            # fall back to the identity mapping if not provided
            self.ensembl_mapping_dict = {k: k for k in self.gene_token_dict.keys()}
        
        self.model.eval()
        self.device=device
        
        if not PREDIFINED_EMB is None:
            self.PREDIFINED_EMB = PREDIFINED_EMB.to(self.device)

    def _tokenize_cell(self, expression_list, gene_list, gene_type='DepMap', target_sum=10000):
        """
        Convert a single cell's expression data into token IDs
        Follows the original logic of Geneformer TranscriptomeTokenizer

        Args:
            gene_list: list of str, list of gene names
            expression_list: list of float, corresponding expression values
            gene_type: str, 'DepMap' (gene symbol) or 'Ensembl'
            target_sum: target total for normalization (default 10000)

        Returns:
            input_ids: list of token IDs, None if there are no valid genes
        """
        # 1. filter out undetected genes (expression > 0)
        # make sure both lists have the same length
        if len(gene_list) != len(expression_list):
            raise ValueError("gene_list and expression_list must have the same length")

        # keep only genes with expression > 0
        detected_genes = []
        detected_exprs = []
        for gene, expr in zip(gene_list, expression_list):
            if expr > 0:
                detected_genes.append(gene)
                detected_exprs.append(expr)

        if len(detected_genes) == 0:
            return None

        # 2. merge genes and obtain tokens and expression
        gene_expr_dict = {}  # {ensembl_id: expression}

        for gene, expr in zip(detected_genes, detected_exprs):
            ensembl_id = None

            # map the gene name to an Ensembl ID
            if gene_type == 'DepMap':
                # DepMap format: "TP53 10" or "TP53"
                gene_symbol = gene.split(' ')[0]
                if gene_symbol in self.gene_symbol_to_ensembl:
                    ensembl_id = self.gene_symbol_to_ensembl[gene_symbol]
                else:
                    continue

            elif gene_type == 'Ensembl':
                # Ensembl format: "ENSG00000141510" or "ENSG00000141510.11"
                raw_ensembl_id = gene.split('.')[0] if '.' in gene else gene
                ensembl_id = raw_ensembl_id

            if ensembl_id is None:
                continue

            # step 1: map/clean the Ensembl ID via ensembl_mapping_dict
            if ensembl_id in self.ensembl_mapping_dict:
                ensembl_id = self.ensembl_mapping_dict[ensembl_id]

            # step 2: merge expression of identical ensembl_ids (sum)
            if ensembl_id in gene_expr_dict:
                gene_expr_dict[ensembl_id] += expr
            else:
                gene_expr_dict[ensembl_id] = expr

        if len(gene_expr_dict) == 0:
            return None

        # step 3: keep only genes that have a token
        valid_ensembl_ids = [eid for eid in gene_expr_dict.keys() if eid in self.gene_token_dict]
        if len(valid_ensembl_ids) == 0:
            return None

        # 4. prepare the arrays needed for normalization
        gene_tokens = []
        expression_values = []
        norm_factors = []

        for ensembl_id in valid_ensembl_ids:
            gene_tokens.append(self.gene_token_dict[ensembl_id])
            expression_values.append(gene_expr_dict[ensembl_id])
            norm_factors.append(self.gene_median_dict.get(ensembl_id, 1.0))

        # 5. normalize expression
        total_counts = sum(expression_values)
        if total_counts == 0:
            return None

        expression_array = np.array(expression_values, dtype=np.float32)
        norm_array = np.array(norm_factors, dtype=np.float32)
        norm_expr = (expression_array / total_counts * target_sum) / norm_array

        # 6. sort descending by normalized expression
        sorted_indices = np.argsort(-norm_expr)
        sorted_tokens = np.array(gene_tokens)[sorted_indices]

        # 7. truncate to the maximum length
        if self.SPECIAL_TOKEN:
            max_seq_len = self.MAX_LENGTH - 2
        else:
            max_seq_len = self.MAX_LENGTH
        sorted_tokens = sorted_tokens[:max_seq_len]

        # 8. add the special tokens
        if self.SPECIAL_TOKEN:
            if self.cls_token_id is None or self.eos_token_id is None:
                raise ValueError("CLS or EOS token ID is None. Check token dictionary.")
            input_ids = [self.cls_token_id] + sorted_tokens.tolist() + [self.eos_token_id]
        else:
            input_ids = sorted_tokens.tolist()

        return input_ids

    def _extract_embedding(self, input_ids, emb_type='cell', emb_layer=-1, device='cuda'):
        """
        Extract embeddings, following the original logic of Geneformer EmbExtractor

        Args:
            input_ids: list of token IDs
            emb_type: str, one of 'cell', 'gene', 'cls'
                      - 'cell': mean-pool the embeddings of all gene tokens
                      - 'cls': CLS token embedding (V2 only)
                      - 'gene': return the embedding of each token
            emb_layer: int, -1: second-to-last layer, 0: last layer
            device: str, device

        Returns:
            emb_type='cell': torch.tensor, shape [hidden_dim]
            emb_type='cls': torch.tensor, shape [hidden_dim]
            emb_type='gene': dict, {token_id: embedding_tensor}
        """
        if input_ids is None:
            return None
        
        original_len = len(input_ids)
        
        # prepare the input: pad to MAX_LENGTH
        if original_len < self.MAX_LENGTH:
            pad_len = self.MAX_LENGTH - original_len
            input_ids_padded = input_ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * original_len + [0] * pad_len
        else:
            input_ids_padded = input_ids[:self.MAX_LENGTH]
            attention_mask = [1] * self.MAX_LENGTH
        
        input_tensor = torch.tensor([input_ids_padded]).to(device)
        attention_mask_tensor = torch.tensor([attention_mask]).to(device)

        # forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor, 
                attention_mask=attention_mask_tensor,
                output_hidden_states=True
            )

        # compute the layer index (original Geneformer logic)
        total_layers = len(outputs.hidden_states)
        if emb_layer == -1:
            # second-to-last layer (2nd to last)
            layer_idx = total_layers - 2
        elif emb_layer == 0:
            # last layer (last)
            layer_idx = total_layers - 1
        else:
            layer_idx = emb_layer

        embeddings = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_dim]
        original_lens = torch.tensor([original_len], device=device)

        # return according to emb_type (following original Geneformer logic)
        if emb_type == 'cls':
            if not self.SPECIAL_TOKEN:
                raise ValueError("CLS token not available when SPECIAL_TOKEN=False (V1 model)")
            # the CLS token is at position 0
            return embeddings[0, 0, :].cpu()

        elif emb_type == 'cell':
            # Cell embedding: mean pooling of gene embeddings
            if self.SPECIAL_TOKEN:
                # V2: exclude CLS (pos 0) and EOS (pos -1)
                # use original length - 2 as the valid length
                valid_len = original_len - 2
                if valid_len <= 0:
                    return None
                gene_embs = embeddings[0, 1:original_len-1, :]
            else:
                # V1: all tokens are gene tokens
                valid_len = original_len
                gene_embs = embeddings[0, :original_len, :]
            
            # mean-pool only over the valid part
            embedding = gene_embs[:valid_len, :].mean(dim=0)
            return embedding.cpu()

        elif emb_type == 'gene':
            # gene embedding: return the embedding of each token
            gene_embeddings = {}
            # return only tokens in the actual sequence (excluding padding)
            for i, token_id in enumerate(input_ids):
                gene_embeddings[token_id] = embeddings[0, i, :].cpu()
            return gene_embeddings

        else:
            raise ValueError(f"emb_type must be 'cell', 'gene', or 'cls', got {emb_type}")
            
    def get_embedding(self, x, z, gene_type='DepMap', target_sum=10000, 
                      emb_type='cell', emb_layer=-1, device='cuda'):
        """
        Obtain cell or gene embeddings
        
        Args:
            x: list, expression values
            z: list, list of gene names
            gene_type: 'DepMap' (gene symbol) or 'Ensembl'
            target_sum: target total for normalization
            emb_type: 'cell', 'gene', 'cls'
            emb_layer: -1 (second-to-last layer) or 0 (last layer)
            device: compute device
        
        Returns:
            corresponding embedding
        """
        self.model.to(self.device)
        input_ids = self._tokenize_cell(x, z, gene_type=gene_type, target_sum=target_sum)
        if input_ids is None:
            return None
        return self._extract_embedding(
            input_ids, 
            emb_type=emb_type, 
            emb_layer=emb_layer, 
            device=self.device
        )
    
    def forward(self, x, gene_type='DepMap',emb_type='cell', emb_layer=-1):
        """
        x is the input: either a tokenized cell line, or [(expression, gene_list), ...]
        """
        if torch.max(x) < len(self.PREDIFINED_EMB) and isinstance(x,torch.Tensor): 
            # if the cell line is in the predefined embeddings and already converted
            # PREDIFINED_EMB has shape [predifined_cell_type_num, D]
            return self.PREDIFINED_EMB[x]
        else:
            emb = []
            for _x in x:
                emb.append(
                    get_embedding(self, _x[0], _x[1], gene_type=gene_type, target_sum=10000, emb_type=emb_type, emb_layer=emb_layer, device=self.device)
                )
            return torch.stack(emb)
        
        
        
            