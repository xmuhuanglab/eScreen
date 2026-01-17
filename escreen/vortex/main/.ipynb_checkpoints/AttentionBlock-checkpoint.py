import torch
import torch.nn as nn
from .RMSNorm import RMSNorm
from .ParallelGatedMLP import ParallelGatedMLP
from .MHA import MHA

class AttentionBlock(nn.Module):
    def __init__(
        self,
        # 基础参数
        hidden_size: int,
        num_attention_heads: int,
        layer_idx: int,
        # 归一化参数
        use_rms_norm: bool = True,
        
        # 投影组参数
        proj_groups: int = 1,
        
        # 数据类型参数
        attn_block_dtype: torch.dtype = torch.bfloat16,
        mlp_dtype: torch.dtype = torch.bfloat16,
        
        # MHA参数
        qkv_proj_bias: bool = True,
        mha_out_proj_bias: bool = True,
        use_flash_attn: bool = True,
        rotary_emb_base: float = 1000000,
        
        # 旋转位置编码参数
        use_interpolated_rotary_pos_emb: bool = False,
        rotary_emb_scaling_factor: float = 1.0,
        
        # GQA参数
        smeared_gqa: bool = False,
        
        # MLP参数 (这些将被传递给ParallelGatedMLP)
        **kwargs
    ) -> None:
        super().__init__()
        
        # 存储基础参数
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        self.proj_groups = proj_groups
        
        # 初始化归一化层
        self.pre_norm = RMSNorm(hidden_size)
        self.post_norm = RMSNorm(hidden_size)
        
        # 初始化计数器
        self.counter = 0
        
        # 初始化MHA
        self.inner_mha_cls = MHA(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            num_heads_kv=num_attention_heads // proj_groups,
            rotary_emb_dim=hidden_size // num_attention_heads,
            qkv_proj_bias=qkv_proj_bias,
            rotary_emb_base=rotary_emb_base,
            causal=True,
            out_proj_bias=mha_out_proj_bias,
            use_flash_attn=use_flash_attn,
        ).to(dtype=attn_block_dtype)

        # 如果需要使用插值的旋转位置编码
        if use_interpolated_rotary_pos_emb:
            swap_mha_rope(
                mha=self.inner_mha_cls,
                kwargs_new_rope={"scaling_factor": rotary_emb_scaling_factor},
            )

        # 如果需要使用smeared GQA
        if smeared_gqa:
            self.inner_mha_cls.num_heads_kv = self.inner_mha_cls.num_heads
            
        # 注册inv_freq缓冲区
        self.inner_mha_cls.rotary_emb.register_buffer("inv_freq", self.inner_mha_cls.rotary_emb.inv_freq)

        # 初始化MLP
        self.mlp = ParallelGatedMLP(
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            inner_size_multiple_of = kwargs.get("inner_size_multiple_of"),
            mlp_activation = kwargs.get("mlp_activation"),
            evo2_style_activations = kwargs.get("evo2_style_activations"),
            model_parallel_size = kwargs.get("model_parallel_size"),
            inner_mlp_size = kwargs.get("inner_mlp_size")
        ).to(dtype=mlp_dtype)

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if type(padding_mask) == torch.Tensor:
            u = u * padding_mask[..., None]

        u = (
            self.inner_mha_cls(
                self.pre_norm(u),
                inference_params=inference_params,
            )
            + u
        )

        if type(padding_mask) == torch.Tensor:
            u = u * padding_mask[..., None]

        u = self.mlp(self.post_norm(u)) + u
        return u#, None