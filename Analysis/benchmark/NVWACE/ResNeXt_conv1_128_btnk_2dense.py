'''
New for ResNeXt:
1. Wider bottleneck
2. Add group for conv2

What this file is
-----------------
The ResNeXt trunk used by the NVWACE ("NvwaCE", ref. 53) baseline in
Supplementary Methods §6.2.5. Upstream, NvwaCE is a ResNeXt-34 pretrained
on 50,040 mouse fetal cis-regulatory profiles and applied to 500 bp one-hot
DNA; here the architecture is instantiated from scratch on 512 bp human
windows and the classifier is replaced by cell-line heads in the training
scripts, exactly as described in §6.2.5. No pretrained NvwaCE weights are
loaded anywhere in this repository.

Nothing external is wrapped -- this is a self-contained torchvision-style
ResNeXt definition, altered in three ways that are load-bearing for the
"1D DNA as a 2D image" trick:

1. ``conv1`` is ``(1, 7)`` with padding ``(0, 3)`` and the stem max-pool is
   ``(1, 3)`` with stride 1, so the length axis is *not* downsampled before
   the residual stages; the only downsampling is the stride-2 in layers 2-4.
2. Every spatial kernel is 1-by-k and the ResNeXt grouped convolution
   (``groups=num_group``) runs along the length axis only.
3. The height axis is a singleton: the expected input is ``(N, 4, 1, L)``,
   one-hot channel axis first, artificial height of 1 second, 512 bp length
   third. ``train_NVWACE.py`` inserts that axis; §6.2.5 reports the same
   ``batch x 4 x 1 x 512`` layout.

Naming: ``conv1_128`` = stem width 128, ``btnk`` = the 1,024 -> 32
bottleneck shipped with the NvwaCE release, ``2dense`` = the two dense
layers (bottleneck + classifier). The classifier built here
(``fc1`` -> ``fc2`` = Sigmoid) is bypassed by both benchmark wrappers and is
kept only because they reuse the trunk.

Expected on disk
----------------
None. The module is import-only (the ``__main__`` block is a shape smoke
test with a random tensor) and loads no weights; ``ResNeXt.load`` accepts a
raw ``state_dict`` path if one is ever available.

Known defects
-------------
* ``ResNeXt.save`` calls ``time.strftime`` but this module never imports
  ``time``, so that helper raises ``NameError``. It is unused here.
* ``resnext50/101/152`` cannot build a working ``fc1`` (see the
  ``expansion`` mismatch documented on ``ResNeXt``); only ``resnext18`` and
  ``resnext34`` run, and §6.2.5 only needs ResNeXt-34.
'''
import torch
import torch.nn as nn
import math

from typing import Callable

__all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101',
           'resnext152']

def conv1x3(in_planes, out_planes, stride=1, groups=32):
    """1x3 convolution with padding.

    Parameters
    ----------
    in_planes : int
        Input channels (4 at the DNA stem).
    out_planes : int
        Output channels; must be divisible by ``groups``.
    stride : int or tuple, default 1
        Forwarded to ``nn.Conv2d``; a stride of 2 halves the length axis.
    groups : int, default 32
        Grouped-convolution width, the ResNeXt cardinality.

    Returns
    -------
    nn.Conv2d
        Kernel ``(1, 3)``, zero padding ``(0, 1)``, no bias (BatchNorm
        supplies the shift).

    Notes
    -----
    Height kernel 1 with height padding 0: this layer never mixes the
    singleton height axis, so it is a plain 1D convolution over the 512 bp
    axis in disguise.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,3), stride=stride,
                     padding=(0,1), bias=False, groups=groups)


class BasicBlock(nn.Module):
    """Two 1x3 grouped convolutions with a residual add (ResNet-18/34 block).

    This is the block used by ``resnext18`` and ``resnext34``, i.e. by the
    ResNeXt-34 configuration of §6.2.5.

    Attributes
    ----------
    expansion : int
        Channel multiplier, 1 here. ``ResNeXt`` uses it to size the next
        stage and the bottleneck.

    Notes
    -----
    * Both convolutions preserve the length axis (same padding); the only
      resolution change comes from the optional ``downsample`` path, which
      carries the stride.
    * ``self.conv1 = conv1x3(inplanes, planes, stride)`` leaves ``groups`` at
      the helper default of 32, so the *first* convolution is grouped as
      well -- unlike the usual ResNet block. Both ``inplanes`` and
      ``planes`` must therefore be divisible by 32, which holds for every
      stage of this network (128 ... 1024).
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        """Initialise the block.

        Parameters
        ----------
        inplanes : int
            Input channels.
        planes : int
            Width of both convolutions (expansion 1).
        stride : int, default 1
            Stride of the first convolution; 2 halves the length axis.
        downsample : nn.Module or None, default None
            Projection applied to the identity path when the shape changes.
        num_group : int, default 32
            Cardinality of the second (grouped) convolution.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Run conv-BN-ReLU-conv-BN, add the residual, then ReLU.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, C, 1, L)``.

        Returns
        -------
        torch.Tensor
            ``(N, planes, 1, L)`` when ``stride == 1``; ``L/2`` otherwise.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """1x1 -> 3x3 grouped -> 1x1 residual block (ResNet-50+ style).

    Present for completeness (``resnext50/101/152``); the benchmark uses
    ``BasicBlock``. As documented on ``ResNeXt``, the ``DenseLayer`` head is
    mis-sized for ``expansion == 4``, so this block currently cannot be used
    end-to-end.

    Attributes
    ----------
    expansion : int
        Channel multiplier, 4 here: the three convolutions are
        ``planes*2``, ``planes*2`` and ``planes*4`` wide.

    Notes
    -----
    The middle 3x3 convolution is the grouped (cardinality) one and carries
    the stride; the two 1x1 convolutions are ungrouped. Because the height
    kernel of ``conv2`` is 3 with padding 1, it does mix the singleton height
    axis -- padded with zeros, so the effect is a per-position channel mix.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        """Initialise the block.

        Parameters
        ----------
        inplanes : int
            Input channels.
        planes : int
            Base width; the three convolutions use ``planes*2``, ``planes*2``
            and ``planes*4`` channels.
        stride : int, default 1
            Stride of the grouped 3x3 convolution.
        downsample : nn.Module or None, default None
            Projection applied to the identity path when the shape changes.
        num_group : int, default 32
            Cardinality of the grouped 3x3 convolution.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Run 1x1 - 3x3(grouped) - 1x1 with BN, add residual, then ReLU.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, C, 1, L)``.

        Returns
        -------
        torch.Tensor
            ``(N, planes * 4, 1, L)``, with ``L`` halved when ``stride == 2``.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DenseLayer(nn.Module):
    """Linear -> BatchNorm -> dropout -> activation, with a singleton insert.

    Used for the NvwaCE bottleneck and for the classifier. The
    ``unsqueeze(1)`` in :meth:`forward` exists only so that
    ``ResNeXt.forward`` can ``flatten(1)`` the result back to ``(N, F)``; it
    carries no information.

    Parameters
    ----------
    in_features : int
        Input width.
    out_features : int
        Output width; also the BatchNorm channel count.
    use_bias : bool, default True
        Forwarded to ``nn.Linear``.
    batch_norm : bool, default True
        When False an ``nn.Identity`` replaces BatchNorm, which also removes
        BatchNorm's train/eval discrepancy on small batches.
    dropout : float, default 0.2
        Dropout probability, applied *before* the activation.
    activation_fn : Callable, default nn.GELU()
        Non-linearity. Note this is a single shared module instance, not a
        factory; ``nn.Identity()`` is passed explicitly where no activation
        is wanted.

    Notes
    -----
    BatchNorm1d sees a ``(N, F)`` tensor, because the ``(N, 1, F)`` view is
    created afterwards; batch statistics are therefore computed over the
    feature axis as intended.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        batch_norm: bool = True,
        dropout: float = 0.2,
        activation_fn: Callable = nn.GELU(),
    ):
        """Initialise the layer; see the class docstring for the arguments."""
        super().__init__()
        self.dense = nn.Linear(in_features, out_features, bias=use_bias)
        self.batch_norm = (
            nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        )
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """Apply the dense block and re-introduce a singleton axis.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, in_features)``.

        Returns
        -------
        torch.Tensor
            ``(N, 1, out_features)``. The extra axis is an artefact:
            callers immediately ``flatten(1)`` it away.

        Notes
        -----
        Order is dense -> BatchNorm -> unsqueeze -> dropout -> activation, i.e.
        dropout is applied to the post-BatchNorm pre-activation.
        """
        x = self.dense(x)
        x = self.batch_norm(x)
        x=x.unsqueeze(1)
        x = self.dropout(x)
        x = self.activation_fn(x)
        return x

class ResNeXt(nn.Module):
    """ResNeXt trunk for 1D DNA plus the NvwaCE bottleneck/classifier.

    Parameters
    ----------
    block : type
        ``BasicBlock`` (expansion 1) or ``Bottleneck`` (expansion 4).
    layers : list of int
        Blocks per stage, e.g. ``[3, 4, 6, 3]`` for ResNeXt-34.
    num_classes : int, default 1000
        Width of the final dense layer. The benchmark passes 5 (one output per
        P0 cell line) and then bypasses the head -- see Notes.
    num_group : int, default 32
        ResNeXt cardinality, forwarded to every grouped convolution.

    Attributes
    ----------
    conv1 : nn.Conv2d
        Stem, 4 -> 128 channels, kernel ``(1, 7)``. The 4 is the DNA one-hot
        axis under the ``(N, 4, 1, L)`` layout of §6.2.5.
    maxpool : nn.MaxPool2d
        ``(1, 3)`` kernel, stride 1, padding ``(0, 1)``: local smoothing only.
    layer1, layer2, layer3, layer4 : nn.Sequential
        Stages with 128/256/512/1024 planes and strides 1/2/2/2.
    bottleneck : DenseLayer
        ``1024 * expansion -> 32`` with BatchNorm, dropout 0.2 and no
        activation.
    fc1 : DenseLayer
        ``32 * expansion -> num_classes``. Correct only when expansion is 1.
    fc2 : nn.Sequential
        A lone ``nn.Sigmoid()``: the original model returned class
        probabilities, not logits.

    Notes
    -----
    * Shape flow for a 512 bp window: ``(N, 4, 1, 512)`` --stem--> 512 --L1-->
      512 --L2--> 256 --L3--> 128 --L4--> 64 --global-avg-pool-->
      ``(N, 1024 * expansion)``.
    * **Known defect for ``Bottleneck``.** ``fc1`` is declared with
      ``in_features = 32 * block.expansion`` while ``bottleneck`` always emits
      32 features. With ``expansion == 1`` the shapes agree; with expansion 4
      ``fc1`` expects 128 and receives 32, so ``resnext50/101/152`` fail on the
      first forward pass. Only ``resnext18``/``resnext34`` are usable, which
      covers §6.2.5.
    * The benchmark wrappers re-implement ``forward`` up to the global average
      pool and attach their own shared layer plus cell-line heads, so
      ``bottleneck``, ``fc1`` and ``fc2`` are dead parameters: they appear in
      the ``state_dict`` but never receive gradients.
    * ``inplanes`` is assigned before ``super().__init__()``; this works but is
      unusual, and it is why the attribute exists as soon as
      ``_make_layer`` runs.
    """

    def __init__(self, block, layers, num_classes=1000, num_group=32):
        """Build the trunk; see the class docstring for the arguments."""
        self.inplanes = 128
        super(ResNeXt, self).__init__()
        # Stem: 4 input channels are the A/C/G/T one-hot of §6.2.5; the kernel
        # is 1x7 so that no mixing happens across the singleton height axis and
        # the 512 bp length is preserved (padding 3 on each side).
        self.conv1 = nn.Conv2d(4, 128, kernel_size=(1,7), stride=1, padding=(0,3),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=1, padding=(0,1))
        # Adaptive pooling instead of the upstream AvgPool2d(7): it makes the
        # trunk independent of the input length, which matters because the
        # benchmark feeds 512 bp windows while the original NvwaCE code used
        # 500 bp ones.
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.layer1 = self._make_layer(block, 128, layers[0], num_group)
        self.layer2 = self._make_layer(block, 256, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], num_group, stride=2)
        
        # NvwaCE bottleneck/classifier. NOTE: ``fc1.in_features`` is written as
        # ``32 * block.expansion`` although the bottleneck always outputs 32, so
        # this pair is only shape-consistent for ``BasicBlock`` (expansion 1).
        self.bottleneck = DenseLayer(
            in_features= 1024 * block.expansion,
            out_features= 32,
            use_bias=True,
            batch_norm=True,
            dropout=0.2,
            activation_fn=nn.Identity(),
        )
        self.fc1 = DenseLayer(
            in_features= 32 * block.expansion,
            out_features= num_classes,
            use_bias=True,
            batch_norm=True,
            dropout=0.2,
            activation_fn=nn.Identity(),
        )

        self.fc2 = nn.Sequential(
          #  nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

        # Kaiming-normal for convolutions (std = sqrt(2/fan_in)), identity for
        # BatchNorm. DenseLayer parameters keep PyTorch defaults -- this loop only
        # matches on Conv2d and BatchNorm2d, so it silently skips them.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        """Stack ``blocks`` residual blocks, inserting a projection if needed.

        Parameters
        ----------
        block : type
            ``BasicBlock`` or ``Bottleneck``.
        planes : int
            Base width of the stage.
        blocks : int
            Number of blocks in the stage.
        num_group : int
            Cardinality passed to every block.
        stride : int, default 1
            Stride of the *first* block only; deeper blocks use stride 1.

        Returns
        -------
        nn.Sequential
            The stage, already accounting for the channel bookkeeping in
            ``self.inplanes`` (which is updated in place).

        Notes
        -----
        A projection is required when the stage changes the channel count or
        downsamples; it is a 1x1 convolution plus BatchNorm. Later blocks are
        constructed with ``self.inplanes`` already set to
        ``planes * block.expansion``, so they see the stage output width.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)
        

    def save(self, fname=None):
        """Persist the state dict to ``fname`` and return the path.

        Parameters
        ----------
        fname : str or None, default None
            Target path. When None a timestamped name is generated.

        Returns
        -------
        str
            The path written.

        Notes
        -----
        **This helper is broken**: it uses ``time.strftime`` but the module
        never imports ``time``, so the ``fname is None`` branch raises
        ``NameError``. Unused by the benchmark scripts, which call
        ``torch.save(model.state_dict(), ...)`` themselves.
        """
        if fname is None:
            fname = time.strftime("model" + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), fname)
        return fname


    def load(self, fpath):
        """Restore weights from a raw ``state_dict`` file.

        Parameters
        ----------
        fpath : str
            Path produced by ``torch.save(model.state_dict(), ...)``.

        Returns
        -------
        None

        Notes
        -----
        Strict by default (``load_state_dict`` default), and the device map is
        not specified, so the checkpoint is restored to the device it was saved
        from.
        """
        self.load_state_dict(torch.load(fpath))


    def forward(self, x):
        """Classify a batch of one-hot DNA windows.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, 4, 1, L)``, or ``(N, 4, L)`` -- the missing height axis is
            inserted here.

        Returns
        -------
        torch.Tensor
            ``(N, num_classes)`` probabilities (``fc2`` ends in a sigmoid).

        Notes
        -----
        The benchmark wrappers do *not* call this method; they inline the
        convolution stack and stop at the global average pool so that they can
        attach per-cell-line heads to the 1,024-d trunk feature.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(2)
        # NOTE: both benchmark wrappers inline this stack and stop after
        # ``avgpool`` to attach their own cell-line heads; keep the two copies in
        # sync if this trunk is ever edited.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x=x.flatten(1)
        x = self.fc1(x)
        x=x.flatten(1)
        x = self.fc2(x)

        return x


def resnext18( **kwargs):
    """Constructs a ResNeXt-18 model.

    Parameters
    ----------
    **kwargs : dict
        Forwarded to :class:`ResNeXt` (``num_classes``, ``num_group``).

    Returns
    -------
    ResNeXt
        Two ``BasicBlock`` per stage, 128 -> 1024 channels.

    Notes
    -----
    This factory is unused by the benchmark, which calls
    ``ResNeXt(BasicBlock, [3, 4, 6, 3], ...)`` directly to get ResNeXt-34
    (§6.2.5).
    """
    model = ResNeXt(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnext34(**kwargs):
    """Constructs a ResNeXt-34 model.

    Parameters
    ----------
    **kwargs : dict
        Forwarded to :class:`ResNeXt`.

    Returns
    -------
    ResNeXt
        ``[3, 4, 6, 3]`` ``BasicBlock`` stages: the NVWACE configuration.

    Notes
    -----
    Equivalent to what ``train_NVWACE.py`` builds inline; kept so that the
    upstream NvwaCE entry point still works.
    """
    model = ResNeXt(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.

    Parameters
    ----------
    **kwargs : dict
        Forwarded to :class:`ResNeXt`.

    Returns
    -------
    ResNeXt
        ``Bottleneck`` stages; see the expansion defect on :class:`ResNeXt`,
        which makes this variant unusable as written.
    """
    model = ResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNeXt-101 model.

    Parameters
    ----------
    **kwargs : dict
        Forwarded to :class:`ResNeXt`.

    Returns
    -------
    ResNeXt
        ``Bottleneck`` stages ``[3, 4, 23, 3]``; same defect as ResNeXt-50.
    """
    model = ResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNeXt-152 model.

    Parameters
    ----------
    **kwargs : dict
        Forwarded to :class:`ResNeXt`.

    Returns
    -------
    ResNeXt
        ``Bottleneck`` stages ``[3, 8, 36, 3]``; same defect as ResNeXt-50.
    """
    model = ResNeXt(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

if __name__ == '__main__':
    model = resnext18(num_classes=5049)
    # model = resnext101(num_classes=5049)
    x = torch.randn(28,4,500)
    out = model(x)
    print(out.shape)
    