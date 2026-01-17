import torch
import torch.nn as nn
import torch.nn.functional as F
from .HyenaInferenceEngine import HyenaInferenceEngine

try:
    from flashfftconv import FlashDepthWiseConv1d
except:
    assert 1==0,'Could not import flashfftconv'


    
    
def interleave(z_pre):
    if len(z_pre.shape) == 3:  # non-cached
        x1 = z_pre[:, 0::3, :]
        x2 = z_pre[:, 1::3, :]
        v = z_pre[:, 2::3, :]
        z_pre = torch.cat([x1, x2, v], dim=1)
        return z_pre
    else:
        x1 = z_pre[..., 0::3]
        x2 = z_pre[..., 1::3]
        v = z_pre[..., 2::3]
        z_pre = torch.concat([x1, x2, v], dim=-1)
        return z_pre
    
    
class HyenaCascade(nn.Module):
    def __init__(
        self,
        # Config parameters
        state_size: int,
        hidden_size: int,
        num_filters: int,
        num_attention_heads: int,
        short_filter_length: int,
        short_filter_bias: bool = False,
        # Optional parameters with defaults
        hyena_filter_groups: int = 1,
        fir_inner_filter_length: int = None,
        layer_idx: int = 0,
        use_flashfft: bool = False,
        inference_mode: bool = True,
        column_split_hyena: bool = True,
        hyena_flip_x1x2: bool = False,
        use_flash_depthwise: bool = False,
        depthwise_dtype: torch.dtype = torch.bfloat16,
        long_fir_threshold: int = None,
        interleave: bool = False,
        prefill_style: str = "fft",
        bidirectional: bool = True
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hyena_filter_groups = hyena_filter_groups

        self.use_flashfft = use_flashfft
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.inference_mode = inference_mode
        self.counter = 0
        self.column_split_hyena = column_split_hyena
        self.hyena_flip_x1x2 = hyena_flip_x1x2

        assert self.hidden_size % self.num_filters == 0 and self.num_filters <= self.hidden_size

        # attention heads are not used except to split post short_filter
        # projections in the same way as the checkpoint
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads

        self.fir_inner_filter_length = fir_inner_filter_length
        self.short_filter_length = short_filter_length
        self.short_filter_weight = nn.Parameter(torch.randn(3 * hidden_size, 1, short_filter_length))
        self.short_filter_bias = nn.Parameter(torch.randn(3 * hidden_size)) if short_filter_bias else None

        self.engine = HyenaInferenceEngine(
            layer_idx=layer_idx,
            hyena_flip_x1x2=hyena_flip_x1x2,
            bidirectional=bidirectional
        )
        self.use_flash_depthwise = use_flash_depthwise
        self.data_dtype = None

        if self.use_flash_depthwise:
            try:
                self.fir_fn = FlashDepthWiseConv1d(
                    channels=3 * self.hidden_size,
                    kernel_size=self.short_filter_length,
                    padding=self.short_filter_length - 1,
                    weights=self.short_filter_weight,
                    bias=self.short_filter_bias,
                    device=None,
                    dtype=depthwise_dtype,
                )
            except ImportError:
                "flashfftconv not installed"
                self.fir_fn = F.conv1d
        else:
            self.fir_fn = F.conv1d
            self.fir_inner_fn = F.conv1d

        self.fftconv_fn = None
        self.long_fir_threshold = long_fir_threshold
        if self.long_fir_threshold is not None:
            assert self.use_flashfft is False, "long_fir_threshold not compatible with fused flashfft"

        self.num_systems = self.hyena_filter_groups
        self.channels_per_group = self.hidden_size // self.hyena_filter_groups
        self.interleave = interleave
        self.prefill_style = prefill_style
        self.bidirectional = bidirectional

        if self.fir_inner_filter_length:
            self.h = nn.Parameter(torch.randn(self.hyena_filter_groups, 1, fir_inner_filter_length))

            if fir_inner_filter_length >= 128:
                self.D = nn.Parameter(torch.zeros(self.hidden_size))
            else:
                self.D = None
        else:
            log_poles = -1*torch.abs(torch.randn(self.num_systems, self.state_size, 1, dtype=torch.float32)) # 这里是初始化
            self.log_poles = nn.Parameter(log_poles) # 这里把log_poles注册为一个可学习的参数
            self.residues = nn.Parameter(torch.randn(self.num_systems, self.state_size, dtype=torch.float32))
            self.D = nn.Parameter(torch.zeros(self.hidden_size))
            self.h = None
        self.t = None

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if inference_params is not None and self.layer_idx in inference_params.fir_state_dict.keys():
            return self.sequential_forward(u, inference_params)
        else:
            return self.parallel_forward(u, inference_params, padding_mask)

    def parallel_forward(self, u, inference_params=None, padding_mask=None):
        L = u.shape[1]
        dims = (
            self.hidden_size,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.state_size,
            self.hyena_filter_groups,
        )

        z_pre, fir_state = self.engine.parallel_fir(
            self.fir_fn,
            u,
            self.short_filter_weight,
            self.short_filter_bias,
            L,
            dims=dims,
            gate=False,
            column_split_hyena=self.column_split_hyena,
            fir_length=self.short_filter_length,
            inference_params=inference_params,
            padding_mask=padding_mask,
            dim_last=True,
            bidirectional = self.bidirectional
        )

        if inference_params:
            inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.interleave:
            z_pre = interleave(z_pre)

        if self.h is None:
            h, _, _, _ = self.compute_filter(L, u.device)
        else:
            h = self.h

        D = self.D

        if self.hyena_filter_groups > 1:
            h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)

        if self.fir_inner_filter_length is not None:

            y, fir_inner_state = self.engine.parallel_fir(
                self.fir_inner_fn,
                z_pre,
                h,
                D,
                L,
                dims=dims,
                gate=True,
                gated_bias=self.fir_inner_filter_length >= 128,
                dim_last=False,
                column_split_hyena=self.column_split_hyena,
                fir_length=self.fir_inner_filter_length,
                inference_params=inference_params,
                padding_mask=padding_mask,
                groups=self.hyena_filter_groups,
                bidirectional=self.bidirectional
            )

            y = y.permute(0, 2, 1)
            if inference_params:
                inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            
            y = self.engine.parallel_iir(
                z_pre,
                h,                       # h在不输入fir_inner_fn的情况下也一定是None
                D,                       # D是大小(hidden_size,)的可学习参数
                L,                       # L是输入序列的长度
                t=self.t,                # 反正t一定是None
                poles=self.log_poles,
                residues=self.residues,
                dims=dims,
                inference_params=inference_params,           # inference_params也一定是None
                layer_idx=self.layer_idx,                    # layer_idx是一个索引,不参与计算
                prefill_style=self.prefill_style,            # 不参与计算
                use_flashfft=self.use_flashfft,              # 不参与计算
                fftconv_fn=self.fftconv_fn,                  # 不参与计算
                column_split_hyena=self.column_split_hyena,  # 不参与计算
                long_fir_threshold=self.long_fir_threshold,  # 不参与计算
                padding_mask=padding_mask,                   # padding_mask也一定是None
            )

        return y, inference_params

    def sequential_forward(self, u, inference_params):
        if self.data_dtype is None:
            self.data_dtype = u.dtype

        if len(u.shape) > 2:
            u = u[:, -1]

        z_pre, fir_state = self.engine.step_fir(
            u,
            inference_params.fir_state_dict[self.layer_idx],
            weight=self.short_filter_weight,
            bias=self.short_filter_bias,
        )
        inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.interleave:
            z_pre = interleave(z_pre)

        x2, x1, v = (
            column_split(z_pre, self.num_attention_heads, self.hidden_size_per_attention_head)
            if self.column_split_hyena
            else z_pre.split([self.hidden_size, self.hidden_size, self.hidden_size], dim=1)
        )

        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        if self.fir_inner_filter_length is not None:
            if self.hyena_filter_groups > 1:
                h = self.h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)
            else:
                h = self.h

            y, fir_inner_state = self.engine.step_fir(
                x1 * v,
                inference_params.fir_inner_state_dict[self.layer_idx],
                weight=h,
                bias=self.D,
                flip_filter=self.fir_inner_filter_length >= 128,
                gated_bias=self.fir_inner_filter_length >= 128,
            )
            y = y * x2
            inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            y, iir_state = self.engine.step_iir(
                x2,
                x1,
                v,
                self.D,
                self.residues,
                self.log_poles,
                inference_params.state_dict[self.layer_idx],
                iir_groups=1,
            )
            inference_params.state_dict[self.layer_idx] = iir_state

        y = y.to(dtype=self.data_dtype)
        return y[:, None], inference_params

    def update_time(self, L, device, max_length=1024):
        if self.t is None:
            self.t = torch.arange(L, device=device)[None, None]
        elif self.t.shape[-1] < L:
            self.t = torch.arange(L, device=device)[None, None]
        else:
            self.t = self.t[..., :L]

    def compute_filter(self, L, device):
        self.update_time(L, device)
        filter_dtype = torch.float32
        residues, log_poles = (
            self.residues.to(filter_dtype),
            self.log_poles.to(filter_dtype),
        )
        
        h = (residues[..., None] * (log_poles * self.t).exp()).sum(1)[None]  # B, D, L ## 这里如果t很大就可能会溢出
        return h, filter_dtype, log_poles, residues