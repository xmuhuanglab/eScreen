"""
Malinois reference architecture (BassetBranched).

What this file is
-----------------
A standalone copy of the Malinois model as published: Malinois is a deep
convolutional network adapted from Basset that scores cell-type-specific CRE
activity from MPRA data (Supplementary Methods §6.2.5). This module contains the
network only — no training loop, no data loading, and no CLI. It is the reference
definition, not the benchmark entry point.

How it relates to the rest of the benchmark
-------------------------------------------
``train_Malinois.py`` and ``train_FACS_Malinois.py`` each define their own variant
of this backbone inline (``BassetBranched5Head`` and ``Basset1Head``) rather than
importing this class. Those variants differ from this one in exactly the ways the
manuscript describes for the adaptation: five cell-line-specific heads instead of
one output unit, ``tanh`` removed, and 512 bp input instead of the MPRA-native
length. Nothing in the repository imports this module, so changes here do not
affect the reported benchmark numbers.

Architecture
------------
Three convolution + batch-norm + ReLU + max-pool stages (4 -> 300 -> 200 -> 200
channels, kernel sizes 19/11/7, pooling 3/4/4), then a flatten and a 1000-unit
fully connected layer with dropout 0.3, and a single ``tanh`` output. With the
default ``input_len=200`` the pooled length is ``200 // 3 // 4 // 4 == 4``, so the
flattened vector is ``200 * 4 == 800`` wide.

Notes
-----
* ``pandas``, ``numpy``, ``optim`` and ``F`` are imported but unused here; they
  were carried over from the training notebook this file was extracted from.
* External dependency: PyTorch only. No pretrained weights are shipped, and no
  file is read from or written to disk. The original Malinois authors distribute
  their own trained weights, but the benchmark retrains from scratch (see
  ``train_Malinois.py``), so nothing is loaded here.
"""
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BassetBranched(nn.Module):
    """Basset-style branched CNN with a single bounded scalar output.

    Parameters
    ----------
    input_len : int, optional
        Length in bp of the one-hot input. Default 200 (the MPRA construct length
        used by the original Malinois work); the eScreen benchmark uses 512.

    Attributes
    ----------
    conv_block : torch.nn.Sequential
        The three shared convolution stages. Channels go 4 -> 300 -> 200 -> 200;
        spatial length shrinks by 3, then 4, then 4.
    fc : torch.nn.Sequential
        Flatten -> ``Linear(200 * pooled_len, 1000)`` -> ReLU -> Dropout(0.3) ->
        ``Linear(1000, 1)`` -> ``tanh``.

    Notes
    -----
    * Padding is "same"-style (``kernel_size // 2`` on each side), so the only
      length reduction comes from the three max-pool layers.
    * ``conv_output_len`` is computed by integer division to match ``MaxPool1d``'s
      floor behaviour; it must be recomputed if ``input_len`` or the pooling
      factors change, otherwise the flatten/linear shapes mismatch at runtime.
    * The ``tanh`` output bounds predictions to [-1, 1], which suits the original
      MPRA regression target but is wrong for logit-based classification — the
      benchmark variants drop it and train with ``BCEWithLogitsLoss``.
    * No softmax/sigmoid: this is a regression head, not a classifier.
    """

    def __init__(self, input_len=200):
        """Build the convolutional trunk and the regression head.

        Parameters
        ----------
        input_len : int, optional
            Input length in bp. Determines the flatten width through
            ``input_len // 3 // 4 // 4``; must match the actual sequence length
            fed to :meth:`forward`.

        Notes
        -----
        The three pooling factors are hard-coded (3, 4, 4) to mirror the layers
        above; if those are changed, this computation must change with them or the
        first ``Linear`` will receive the wrong number of features.
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(4, 300, kernel_size=19, padding=9),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(300, 200, kernel_size=11, padding=5),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(200, 200, kernel_size=7, padding=3),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        conv_output_len = input_len
        conv_output_len = conv_output_len // 3  # First pool
        conv_output_len = conv_output_len // 4  # Second pool
        conv_output_len = conv_output_len // 4  # Third pool

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200 * conv_output_len, 1000),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1000, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            One-hot DNA, shape ``(B, 4, input_len)`` — channels first, as
            required by ``Conv1d``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)``, ``tanh``-bounded scalar activity per sequence.

        Notes
        -----
        The output keeps the trailing dimension (no ``squeeze``), so a BCE-style
        loss would need ``.squeeze(-1)`` — one more reason the benchmark variants
        define their own heads.
        """
        x = self.conv_block(x)
        x = self.fc(x)
        return x