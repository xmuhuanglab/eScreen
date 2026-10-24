"""
Sei architecture

Vendored copy of the Sei architecture (Chen et al., Nature Genetics 2022), the
sequence-to-regulatory-activity model that maps a 4,096 bp one-hot window to
21,907 chromatin tracks. It is reproduced here so that Sei can be run as a
baseline on the same 512 bp CRE dataset as eScreen.

Scope: **this file is not described anywhere in the manuscript.**
Supplementary Methods §6.2 covers Enformer (§6.2.1), AlphaGenome (§6.2.2),
HyenaDNA (§6.2.3), LegNet/Dream-CNN/Dream-RNN (§6.2.4) and
DeepSEA/NvwaCE/Malinois (§6.2.5); there is no Sei section, so none of the layer
widths, dilations or spline settings below can be attributed to the paper.
They are the upstream Sei defaults.

What is actually used
---------------------
* ``bs`` / ``spline_factory`` / ``BSplineTransformation`` -- the cubic B-spline
  basis that compresses the length axis from 64 positions to 16 coefficients
  per channel. Both training scripts import ``BSplineTransformation`` from
  here.
* ``BSplineConv1D`` -- the spline-convolution building block of the original
  Sei code. Unused by the benchmark, which uses plain ``nn.Conv1d`` for the
  trunk.
* ``Sei_Default`` -- the complete 21,907-output classifier. **Broken and
  unused**: its constructor calls ``super(Sei, self).__init__()`` while the
  class is named ``Sei_Default``, so instantiating it raises ``NameError``.
  The training scripts therefore re-implement the trunk inline and the two
  copies must be kept in sync by hand.
* ``criterion`` / ``get_optimizer`` -- upstream training hooks (BCELoss, SGD
  with momentum 0.9). The benchmark uses AdamW and never calls them.

Expected on disk
----------------
Nothing. No Sei pretrained weights are downloaded or loaded anywhere in this
repository: the baseline is trained from scratch on the eScreen data.

Coordinate conventions
----------------------
Everything assumes channels-first 1-D tensors ``(N, C, L)`` and a one-hot base
order of A, C, G, T. ``sequence_length`` defaults to 4,096 only because that is
Sei's native input width; the benchmark centre-pads 512 bp windows up to it.
"""
import numpy as np
import torch
import torch.nn as nn



import numpy as np
from scipy.interpolate import splev


def bs(x, df=None, knots=None, degree=3, intercept=False):
    """
    df : int
        The number of degrees of freedom to use for this spline. The
        return value will have this many columns. You must specify at least
        one of `df` and `knots`.
    knots : list(float)
        The interior knots of the spline. If unspecified, then equally
        spaced quantiles of the input data are used. You must specify at least
        one of `df` and `knots`.
    degree : int
        The degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept : bool
        If `True`, the resulting spline basis will span the intercept term
        (i.e. the constant function). If `False` (the default) then this
        will not be the case, which is useful for avoiding overspecification
        in models that include multiple spline terms and/or an intercept term.

    Parameters
    ----------
    x : array_like, shape (n,)
        Sample locations (here always a 0..L-1 position index, or a signed
        log-distance in the ``log=True`` branch of ``spline_factory``).
    df : int or None, default None
        Degrees of freedom, i.e. the number of returned columns. Mutually
        exclusive with ``knots``; at least one of the two must be given.
    knots : list of float or None, default None
        Interior knots. When None, ``df - order + (1 - intercept)`` interior
        knots are placed at equally spaced percentiles of ``x``, so the basis
        adapts to the sampling density of ``x``.
    degree : int, default 3
        Piecewise-polynomial degree (3 = cubic).
    intercept : bool, default False
        If True the basis spans the constant function and every column is
        kept; if False the first column is dropped to avoid collinearity with
        an explicit intercept term.

    Returns
    -------
    numpy.ndarray, shape (n, n_basis)
        B-spline design matrix, evaluated with ``scipy.interpolate.splev`` one
        unit coefficient vector at a time.

    Notes
    -----
    * The boundary knots are ``min(x)`` and ``max(x)`` repeated ``order``
      times (a clamped spline), so the basis is well defined on
      ``[min(x), max(x)]``; positions outside that interval extrapolate.
    * A too-small ``df`` is clamped to ``order - (1 - intercept)`` with a
      printed message instead of raising, which can silently return fewer
      columns than the caller asked for.
    * Cost is ``n_basis`` calls to ``splev``, i.e. O(n * df); this is why the
      basis is built once and cached in ``BSplineTransformation``.

    """

    order = degree + 1
    inner_knots = []
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - intercept)
        if n_inner_knots < 0:
            n_inner_knots = 0
            print("df was too small; have used %d"
                  % (order - (1 - intercept)))

        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1])

    elif knots is not None:
        inner_knots = knots

    all_knots = np.concatenate(
        ([np.min(x), np.max(x)] * order, inner_knots))

    all_knots.sort()

    n_basis = len(all_knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_basis), dtype=float)

    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        basis[:, i] = splev(x, (all_knots, coefs, degree))

    if not intercept:
        basis = basis[:, 1:]
    return basis


def spline_factory(n, df, log=False):
    """Build the B-spline design matrix used by the Sei layers.

    Parameters
    ----------
    n : int
        Number of input positions (length of the sequence axis).
    df : int
        Number of spline coefficients to produce per channel.
    log : bool, default False
        If True, evaluate the basis on the signed log-distance axis
        ``sign(i - n/2) * log(|i - n/2| + 1)`` instead of the raw index, which
        is symmetric about the window centre and compresses the tails.

    Returns
    -------
    torch.Tensor
        ``float32`` matrix ``(n, df)``; multiplying ``(N, C, n)`` by it gives
        ``(N, C, df)``.

    Notes
    -----
    The log branch uses ``df - 4`` interior knots spread evenly over the
    log-distance range, so ``df`` must be at least 4 there; the linear branch
    forwards ``df`` to :func:`bs` unchanged.
    """
    if log:
        dist = np.array(np.arange(n) - n/2.0)
        dist = np.log(np.abs(dist) + 1) * ( 2*(dist>0)-1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist),np.max(dist),n_knots+2)[1:-1]
        return torch.from_numpy(bs(
            dist, knots=knots, intercept=True)).float()
    else:
        dist = np.arange(n)
        return torch.from_numpy(bs(
            dist, df=df, intercept=True)).float()



class BSplineTransformation(nn.Module):

    """Fixed linear map from ``L`` positions to ``df`` spline coefficients.

    Multiplies ``(N, C, L)`` by a cached ``(L, df)`` B-spline basis, giving
    ``(N, C, df)``. In the Sei trunk this takes the 64-position feature map
    (three 4x max-pools of a 4,096 bp input) down to 16 coefficients per
    channel, i.e. a smooth, low-dimensional summary of the profile.

    Parameters
    ----------
    degrees_of_freedom : int
        Number of output coefficients per channel (16 in the trunk).
    log : bool, default False
        Build the basis on a signed log-distance axis instead of a plain
        position index.
    scaled : bool, default False
        Divide the basis by the input length. The trunk passes False.

    Notes
    -----
    * The basis is cached in a plain attribute rather than a registered
      buffer: it is not moved by ``.to(device)``, does not appear in the
      ``state_dict``, and is created lazily on the first forward pass.
    * The cache is never invalidated, so calling the module later with a
      different length silently reuses the stale basis.
    """

    def __init__(self, degrees_of_freedom, log=False, scaled=False):
        """Configure the transformation; the basis itself is built lazily.

        Parameters
        ----------
        degrees_of_freedom : int
            Output coefficients per channel.
        log : bool, default False
            Use the signed log-distance axis.
        scaled : bool, default False
            Divide the basis by the input length.
        """
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._log = log
        self._scaled = scaled
        self._df = degrees_of_freedom

    def forward(self, input):
        """Project the length axis onto the spline basis.

        Parameters
        ----------
        input : torch.Tensor
            ``(N, C, L)``; the basis is built for the ``L`` of the first call and
            cached for every later call.

        Returns
        -------
        torch.Tensor
            ``(N, C, degrees_of_freedom)``.

        Notes
        -----
        The cached basis is moved to ``input.device`` only on the call that
        creates it. Moving the module afterwards (or changing ``L``) leaves a
        stale basis and raises a device or shape error -- an accepted limitation
        of the vendored code.
        """
        if self._spline_tr is None:
            spatial_dim = input.size()[-1]
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim
            if input.is_cuda:
                self._spline_tr = self._spline_tr.to(input.device)
        
        return  torch.matmul(input, self._spline_tr)



class BSplineConv1D(nn.Module):

    """Spline smoothing followed by a trainable 1x1 channel mixer.

    Upstream Sei building block, unused by this benchmark. A frozen
    ``Conv1d(1, df, kernel_size)`` whose weights *are* the B-spline basis does
    the smoothing; a trainable ``Conv1d(C*df, out, 1)`` mixes channels. To
    apply the single-channel filter to every input channel at once the input is
    reshaped to ``(N*C, 1, L)``.

    Parameters
    ----------
    in_channels : int
        Input channel count.
    out_channels : int
        Output channel count.
    kernel_size : int
        Length of the spline filter; also the number of basis functions, one
        per kernel position.
    degrees_of_freedom : int
        Number of spline coefficients produced per input channel.
    stride, padding, dilation : int, default 1, 0, 1
        Applied to the frozen spline convolution only.
    groups : int, default 1
        Groups of the trainable 1x1 convolution, whose input width is
        ``in_channels * degrees_of_freedom``.
    bias : bool, default True
        Bias of the trainable convolution.
    log : bool, default False
        Build the spline basis on the signed log-distance axis.
    scaled : bool, default True
        Normalise the spline weights by ``kernel_size``.

    Notes
    -----
    The spline weights are assigned as a plain tensor before being wrapped in
    ``nn.Parameter`` with ``requires_grad = False``; the intermediate state is
    not a valid parameter assignment, and the code only works because the
    parameter wrapper is restored immediately afterwards.
    """

    def __init__(self, in_channels, out_channels, kernel_size, degrees_of_freedom, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, log=False, scaled = True):
        """Build the frozen spline filter and the trainable 1x1 mixer.

        See the class docstring for the parameters. The spline weights are
        frozen; only the trailing ``conv1d`` is optimised.
        """
        super(BSplineConv1D, self).__init__()
        self._df = degrees_of_freedom
        self._log = log
        self._scaled = scaled

        self.spline = nn.Conv1d(1, degrees_of_freedom, kernel_size, stride, padding, dilation,
            bias=False)
        # The spline filter is deterministic: it *is* the B-spline basis and is
        # never trained. ``view(df, 1, kernel_size)`` turns it into a (1, k) filter
        # applied to each input channel independently.
        self.spline.weight = spline_factory(kernel_size, self._df, log=log).view(self._df, 1, kernel_size)
        if scaled:
            self.spline.weight = self.spline.weight / kernel_size            
        self.spline.weight = nn.Parameter(self.spline.weight)
        self.spline.weight.requires_grad = False
        self.conv1d = nn.Conv1d(in_channels * degrees_of_freedom, out_channels, 1, 
            groups = groups, bias=bias)

    def forward(self, input):
        """Smooth each channel with the spline filter, then mix channels.

        Parameters
        ----------
        input : torch.Tensor
            ``(N, C_in, L)``.

        Returns
        -------
        torch.Tensor
            ``(N, C_out, L')``, with ``L'`` set by the spline convolution's
            stride/padding/dilation.

        Notes
        -----
        The reshape to ``(N*C, 1, L)`` applies the same single-channel filter to
        every input channel; the reshape back to ``(N, C*df, L')`` concatenates
        each channel's coefficients along the channel axis, which is what the
        1x1 mixer consumes.
        """
        batch_size, n_channels, length = input.size()
        spline_out = self.spline(input.view(batch_size * n_channels,1,length))
        conv1d_out = self.conv1d(spline_out.view(batch_size, n_channels * self._df,  length))
        return conv1d_out


class Sei_Default(nn.Module):

    """Full Sei classifier: 21,907 chromatin tracks from 4,096 bp of DNA.

    Three ``lconv``/``conv`` stages (4 -> 480 -> 640 -> 960 channels, with 4x
    max-pooling from the second stage on), five dilated residual convolutions
    whose dilation doubles up to 25, a spline compression to 16 coefficients,
    and a two-layer classifier ending in a sigmoid.

    Parameters
    ----------
    sequence_length : int, default 4096
        Expected input width. It sizes no layer directly (all convolutions are
        length-agnostic) but sets the number of spline coefficients.
    n_genomic_features : int, default 21907
        Number of output tracks (the full Sei panel).

    Notes
    -----
    **Broken**: the constructor calls ``super(Sei, self).__init__()`` but no
    ``Sei`` class exists, so instantiation raises ``NameError``. The class is
    kept as the reference definition of the trunk that the training scripts
    re-implement inline.
    """
    def __init__(self, sequence_length=4096, n_genomic_features=21907):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        # BUG retained from the vendored source: ``Sei`` is undefined -- the class
        # is named ``Sei_Default`` and no module-level name ``Sei`` exists.
        # Instantiating this class therefore raises NameError, which is why the
        # benchmark trains an inline copy of the trunk instead.
        super(Sei, self).__init__()

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4))

        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4))

        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9,padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9,padding=4),
            nn.ReLU(inplace=True))

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4))

        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9,padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9,padding=4),
            nn.ReLU(inplace=True))

        self.dconv1 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8),
            nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16),
            nn.ReLU(inplace=True))
        self.dconv4 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32),
            nn.ReLU(inplace=True))
        self.dconv5 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50),
            nn.ReLU(inplace=True))

        # 4,096 bp -> three 4x max-pools -> 64 positions -> 16 spline coefficients
        # per channel, so the classifier sees 960 * 16 = 15,360 features.
        self._spline_df = 16        
        self.spline_tr = nn.Sequential(
            nn.Dropout(p=0.5),
            BSplineTransformation(self._spline_df, scaled=False))
        #self.norm = nn.BatchNorm1d(num_features=960 * self._spline_df)
        self.classifier = nn.Sequential(
            nn.Linear(960 * self._spline_df, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid()
        )



    def forward(self, x):
        """Forward propagation of a batch.

        Parameters
        ----------
        x : torch.Tensor
            One-hot DNA ``(N, 4, 4096)``, channels first, A/C/G/T order.

        Returns
        -------
        torch.Tensor
            ``(N, n_genomic_features)`` sigmoid probabilities.

        Notes
        -----
        The skips are not the usual ResNet ones: each ``conv`` stage adds the
        *pre-pooling* output of the matching ``lconv`` stage (``out1 + lout1``),
        and the dilated stack propagates a running sum
        (``cat_outN = cat_out(N-1) + dconvN``), so the final feature map is the
        sum of all five dilation scales.
        """
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)

        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)

        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)

        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        # Final feature map: running sum over the five dilation scales, i.e. an
        # effective receptive field of roughly 1 + 2*(2+4+8+16+25)*4 = 441 bp.
        out = cat_out4 + dconv_out5
        
        spline_out = self.spline_tr(out)
        reshape_out = spline_out.view(spline_out.size(0), 960 * self._spline_df)
        predict = self.classifier(reshape_out)
        return predict

def criterion():
    """
    The criterion the model aims to minimize.

    Returns
    -------
    torch.nn.Module
        ``nn.BCELoss`` -- note it expects *probabilities*, which is why
        ``Sei_Default`` ends in a sigmoid.

    Notes
    -----
    Upstream helper, unused by the benchmark scripts, which prefer
    ``BCEWithLogitsLoss`` on logits for numerical stability.
    """
    return nn.BCELoss()

def get_optimizer(lr):
    """
    The optimizer and the parameters with which to initialize the optimizer.
    At a later time, we initialize the optimizer by also passing in the model
    parameters (`model.parameters()`). We cannot initialize the optimizer
    until the model has been initialized.

    Parameters
    ----------
    lr : float
        Learning rate.

    Returns
    -------
    tuple
        ``(torch.optim.SGD, kwargs)`` rather than an optimizer instance, so the
        caller still has to pass ``model.parameters()``.

    Notes
    -----
    Upstream helper, unused: the benchmark scripts train this architecture with
    AdamW (see ``train_Sei.py``) and never call this factory.
    """
    return (torch.optim.SGD, {"lr": lr, "weight_decay": 1e-7, "momentum": 0.9})