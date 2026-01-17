import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .HyenaCascade import HyenaCascade
from .RMSNorm import RMSNorm
from .ParallelGatedMLP import ParallelGatedMLP


class ParallelGatedConvBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        qkv_proj_bias: bool,
        hyena_out_proj_bias: bool,
        hyena_filter_groups: Optional[int] = None,
        fir_inner_filter_length: Optional[int] = None,
        low_mem_mode: bool = False,
        hyena_block_dtype: torch.dtype = torch.bfloat16,
        mlp_dtype: torch.dtype = torch.bfloat16,
        use_fp8_input_projections: bool = False,
        compile: bool = False,
        *args,   # 没有出现在上面的位置参数
        **kwargs # 没有出现在上面的关键字参数
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.low_mem_mode = low_mem_mode
        self.fir_inner_filter_length = fir_inner_filter_length
        self.hyena_filter_groups = hyena_filter_groups if hyena_filter_groups is not None else hidden_size
        
        self.pre_norm, self.post_norm = (
            RMSNorm(hidden_size).to(dtype=hyena_block_dtype),
            RMSNorm(hidden_size).to(dtype=hyena_block_dtype),
        )
        
        self.filter = HyenaCascade(
            state_size =kwargs.get("state_size"),
            hidden_size=hidden_size,
            num_filters=kwargs.get("num_filters"),
            num_attention_heads=kwargs.get("num_attention_heads"),
            short_filter_length=kwargs.get("short_filter_length"),
            short_filter_bias  =kwargs.get("short_filter_bias"),
            hyena_filter_groups=self.hyena_filter_groups,
            fir_inner_filter_length=fir_inner_filter_length,
            layer_idx=layer_idx,
            use_flashfft        = kwargs.get("use_flashfft"),
            inference_mode      = kwargs.get("inference_mode"),
            column_split_hyena  = kwargs.get("column_split_hyena"),
            hyena_flip_x1x2     = kwargs.get("hyena_flip_x1x2"),
            use_flash_depthwise = kwargs.get("use_flash_depthwise"),
            depthwise_dtype     = kwargs.get("depthwise_dtype"),
            long_fir_threshold  = kwargs.get("long_fir_threshold"),
            interleave          = kwargs.get("interleave"),
            prefill_style       = kwargs.get("prefill_style"),
            bidirectional       = kwargs.get("bidirectional")
        ).to(dtype=hyena_block_dtype)

        self.projections = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_proj_bias).to(dtype=hyena_block_dtype)
        self.out_filter_dense = nn.Linear(hidden_size, hidden_size, bias=hyena_out_proj_bias).to(dtype=hyena_block_dtype)
        self.mlp = ParallelGatedMLP(
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            inner_size_multiple_of = kwargs.get("inner_size_multiple_of"),
            mlp_activation = kwargs.get("mlp_activation"),
            evo2_style_activations = kwargs.get("evo2_style_activations"),
            model_parallel_size = kwargs.get("model_parallel_size"),
            inner_mlp_size = kwargs.get("inner_mlp_size")
        ).to(dtype=mlp_dtype)

        if compile:
            self.proj_norm_fn = torch.compile(self.proj_norm, fullgraph=True, dynamic=False, mode="reduce-overhead")
            self.res_mlp_norm_fn = torch.compile(
                self.res_mlp_norm, fullgraph=True, dynamic=False, mode="reduce-overhead"
            )
        else:
            self.proj_norm_fn = self.proj_norm
            self.res_mlp_norm_fn = self.res_mlp_norm
            
        self.use_fp8_input_projections = use_fp8_input_projections

    def pad_to_multiple(self, x, multiple=16):
        """Pad input tensor to multiple of 16 only when FP8 is enabled"""
        if not self.use_fp8_input_projections:
            return x

        batch_size, seq_len, hidden_dim = x.size()
        pad_len = (multiple - (seq_len % multiple)) % multiple
        if pad_len == 0:
            return x
        return F.pad(x, (0, 0, 0, pad_len))

    def proj_norm(self, x):

        normalized = self.pre_norm(x)
        normalized = self.pad_to_multiple(normalized)
        with torch.cuda.device(x.device):
            projected = self.projections(normalized)

        if isinstance(projected, tuple):
            projected = projected[0]

        original_seq_len = x.size(1)
        if projected.size(1) > original_seq_len:
            projected = projected[:, :original_seq_len, :]

        return projected

    def res_mlp_norm(self, x):
        return self.mlp(self.post_norm(x)) + x

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        z = self.proj_norm_fn(u)

        if type(padding_mask) == torch.Tensor:
            z = z * padding_mask[..., None]

        z, inference_params = self.filter(z, inference_params=inference_params, padding_mask=padding_mask)

        z_in = self.out_filter_dense(z) + u

        if type(padding_mask) == torch.Tensor:
            z_in = z_in * padding_mask[..., None]

        y = self.res_mlp_norm_fn(z_in)

        return y#, inference_params