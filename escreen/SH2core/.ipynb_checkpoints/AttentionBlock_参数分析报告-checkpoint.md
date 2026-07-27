# AttentionBlock 参数分析报告

## 一、AttentionBlock 结构概览

`AttentionBlock` 是典型的 Transformer Block 实现，包含以下组件：

- **RMSNorm** 归一化层（pre_norm + post_norm）
- **MHA** 多头注意力机制（含 FlashAttention 支持）
- **ParallelGatedMLP** 门控 MLP
- **残差连接**（每个子层后都有 +input）

---

## 二、全部参数详解

### 1. 基础参数

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `hidden_size` | int | 必填 | 模型隐藏层维度，决定整个块的特征维度。直接影响参数量、显存和计算量 |
| `num_attention_heads` | int | 必填 | 注意力头数。头数越多，特征提取能力越强，但计算量越大 |
| `layer_idx` | int | 必填 | 当前层索引，用于 MLP 激活函数的选择（evo2 风格） |

### 2. 归一化参数

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `use_rms_norm` | bool | True | 是否使用 RMSNorm。该参数仅控制是否使用，代码中已硬编码为 True，实际无法改变 |

### 3. 投影组参数

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `proj_groups` | int | 1 | 投影分组数。用于分组注意力（Grouped Query Attention）。增大此值可减少 KV 头的数量，从而减少显存和计算量 |

### 4. 数据类型参数

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `attn_block_dtype` | torch.dtype | torch.bfloat16 | 注意力块的计算精度。使用 bfloat16 可显著减少显存占用（相比 fp32 节省约一半） |
| `mlp_dtype` | torch.dtype | torch.bfloat16 | MLP 的计算精度。同样 bfloat16 可节省显存 |

### 5. MHA（多头注意力）相关参数

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `qkv_proj_bias` | bool | True | QKV 投影是否加偏置。去掉偏置可减少少量参数 |
| `mha_out_proj_bias` | bool | True | 注意力输出投影是否加偏置。去掉偏置可减少参数 |
| `use_flash_attn` | bool | True | 是否使用 FlashAttention。启用后推理速度更快，显存更优 |
| `rotary_emb_base` | float | 1000000 | RoPE 的基准频率。影响位置编码的感受野 |

### 6. 旋转位置编码参数

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `use_interpolated_rotary_pos_emb` | bool | False | 是否使用插值 RoPE。允许缩放位置编码以处理更长序列 |
| `rotary_emb_scaling_factor` | float | 1.0 | RoPE 缩放因子。1.0 表示无缩放；增大可扩展有效序列长度 |

### 7. GQA 参数

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `smeared_gqa` | bool | False | 是否使用 smeared GQA。启用后会让 num_heads_kv = num_heads，保持全头数，不减少计算量 |

### 8. MLP 相关参数（通过 kwargs 传递）

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `inner_size_multiple_of` | int | 64 | MLP 内层维度的对齐要求。64 的倍数便于 GPU 加速 |
| `mlp_activation` | str | gelu | MLP 激活函数。可选 gelu 或 silu |
| `evo2_style_activations` | bool | False | evo2 风格激活。>0 层时跳过激活函数 |
| `model_parallel_size` | int | 1 | 模型并行大小 |
| `inner_mlp_size` | int | None | 直接指定 MLP 内层维度 |

---

## 三、小参数/小显存配置建议

若以**较小参数量和显存占用**为目标，同时保持不错效果，推荐以下设置：

### 核心策略

**1. 降低 hidden_size**
这是最直接影响参数量的参数。hidden_size 从 4096 降到 2048，参数量可减少约 60-75%。

**2. 减少注意力头数 (num_attention_heads)**
头数减少意味着 head_dim 增大（如果 hidden_size 不变），或整体计算量下降。建议使用 8-16 个头而非 64+ 个。

**3. 启用 proj_groups（GQA 策略）**
增大 proj_groups，使 KV 头数远少于 Q 头数。例如 proj_groups=4 或 8，可减少约 75-87.5% 的 KV 计算量。

**4. 使用 FlashAttention**
`use_flash_attn=True` 能显著降低显存占用并加速推理。

**5. 使用 bfloat16**
`attn_block_dtype=torch.bfloat16` 和 `mlp_dtype=torch.bfloat16` 确保使用低精度。

**6. 去掉不必要的偏置**
`qkv_proj_bias=False` 和 `mha_out_proj_bias=False` 可进一步减少参数。

**7. 避免使用 smeared_gqa**
`smeared_gqa=False` 保持 GQA 模式，让 KV 头数减少。

### 推荐配置示例

```python
AttentionBlock(
    hidden_size=2048,           # 原来可能是 4096，减少 60%+
    num_attention_heads=16,     # 原来可能是 64，减少头数
    layer_idx=0,
    proj_groups=4,              # GQA模式：16个Q头，4个KV头
    attn_block_dtype=torch.bfloat16,
    mlp_dtype=torch.bfloat16,
    use_flash_attn=True,        # 启用FlashAttention
    qkv_proj_bias=False,        # 去掉偏置
    mha_out_proj_bias=False,    # 去掉偏置
    use_interpolated_rotary_pos_emb=False,
    rotary_emb_base=1000000,
    smeared_gqa=False,          # 保持GQA模式
)
```

### 参数量估算

假设原始配置为 hidden_size=4096, num_heads=64, proj_groups=1：

| 配置对比 | 参数量变化 | 显存变化 | 计算量变化 |
|----------|-----------|---------|-----------|
| hidden_size 4096→2048 | ~75%↓ | ~60%↓ | ~75%↓ |
| num_heads 64→16 | 同上 | 同上 | ~75%↓ |
| proj_groups 1→4 (GQA) | 少量↓ | ~75%↓ KV缓存 | ~75%↓ KV计算 |
| bfloat16 vs fp32 | 无变化 | ~50%↓ | 无变化 |
| 去掉 bias | ~1%↓ | 无变化 | 无变化 |

组合以上所有策略，相比原始配置，预计可减少 **70-80%** 的参数量和显存占用，同时保持合理的模型表达能力。

### 注意事项

- 减少参数量可能导致表达能力下降，需通过验证集测试效果
- GQA 模式在大多数情况下性能损失极小，是性价比最高的优化手段
- FlashAttention 需要安装 flash-attn 包，且仅支持 CUDA
- bfloat16 需要 A100/H100 等较新的 GPU 支持；否则使用 fp16
- 如果显存仍然紧张，可考虑使用 `checkpointing=True` 在 MHA 中进行梯度检查点（checkpointing 参数在 MHA 中可用）

---

*生成时间: 2026-04-21*