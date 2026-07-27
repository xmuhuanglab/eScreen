# ParallelGatedConvBlock 参数分析报告

## 一、模块结构概览

`ParallelGatedConvBlock` 是 HyenaDDP 架构中的核心计算块，结合了：
- **HyenaCascade**（Hyena 滤波/序列建模，替代传统注意力）
- **ParallelGatedMLP**（门控 MLP）
- **RMSNorm** 归一化
- **线性投影**（projections + out_filter_dense）
- **残差连接**

---

## 二、ParallelGatedConvBlock 自身参数详解

### 1. 基础参数

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `hidden_size` | int | 必填 | 隐藏层维度，决定整个块的特征维度。直接影响所有子模块的计算量和显存 |
| `layer_idx` | int | 必填 | 当前层索引，用于 Hyena 滤波器的层索引定位 |
| `qkv_proj_bias` | bool | 必填 | QKV 投影线性层是否加偏置。False 可减少少量参数 |
| `hyena_out_proj_bias` | bool | 必填 | Hyena 输出投影是否加偏置。False 可减少参数 |

### 2. 直接传入 HyenaCascade 的参数

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `hyena_filter_groups` | int | hidden_size | 滤波分组数。分组越多，各组的通道数越少，显存越小。默认等于 hidden_size（每组1通道） |
| `hyena_block_dtype` / hyena 相关 dtype | torch.dtype | torch.bfloat16 | 计算精度，bf16 比 fp32 省一半显存 |
| `low_mem_mode` | bool | False | 低内存模式开关 |

### 3. 传递给 HyenaCascade 的参数（通过 kwargs）

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `state_size` | int | None | Hyena 滤波器的状态维度。控制 IIR 滤波器的极点/留数数量 |
| `num_filters` | int | None | 滤波器数量。影响滤波器的表达能力 |
| `num_attention_heads` | int | None | 注意力头数（此处仅用于拆分短滤波器投影） |
| `short_filter_length` | int | None | 短滤波器长度。控制 FIR 短滤波器的感受野 |
| `short_filter_bias` | bool | None | 短滤波器是否加偏置 |
| `fir_inner_filter_length` | int | None | 内层 FIR 滤波器长度。如果指定则启用内层滤波 |
| `use_flashfft` | bool | None | 是否使用 FlashFFT。加速 FFT 卷积 |
| `inference_mode` | bool | None | 推理模式开关 |
| `column_split_hyena` | bool | None | Hyena 列拆分模式 |
| `hyena_flip_x1x2` | bool | None | 翻转 x1/x2 顺序 |
| `use_flash_depthwise` | bool | None | 是否使用 FlashDepthwiseConv1d |
| `depthwise_dtype` | torch.dtype | None | 深度可分离卷积的精度 |
| `long_fir_threshold` | int | None | 长 FIR 阈值 |
| `interleave` | bool | None | 交错模式 |
| `prefill_style` | str | fft | 预填充风格（fft 或其他） |
| `bidirectional` | bool | True | 是否双向滤波 |

### 4. MLP 相关参数（通过 kwargs 传递给 ParallelGatedMLP）

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `inner_size_multiple_of` | int | 64 | MLP 内层维度对齐要求 |
| `mlp_activation` | str | gelu | MLP 激活函数（gelu/silu） |
| `evo2_style_activations` | bool | False | evo2 风格激活 |
| `model_parallel_size` | int | 1 | 模型并行大小 |
| `inner_mlp_size` | int | None | 直接指定 MLP 内层维度 |

### 5. 其他参数

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `hyena_block_dtype` | torch.dtype | torch.bfloat16 | AttentionBlock 的计算精度 |
| `mlp_dtype` | torch.dtype | torch.bfloat16 | MLP 的计算精度 |
| `use_fp8_input_projections` | bool | False | 是否使用 FP8 输入投影（需要特殊 GPU） |
| `compile` | bool | False | 是否使用 torch.compile 编译 |

---

## 三、HyenaCascade 详细参数（被 ParallelGatedConvBlock 调用）

| 参数名 | 类型 | 默认值 | 意义 |
|--------|------|--------|------|
| `state_size` | int | 必填 | IIR 滤波器状态维度。决定极点/留数的数量 |
| `hidden_size` | int | 必填 | 隐藏层维度 |
| `num_filters` | int | 必填 | 滤波器数量 |
| `num_attention_heads` | int | 必填 | 注意力头数 |
| `short_filter_length` | int | 必填 | 短滤波器长度 |
| `short_filter_bias` | bool | False | 短滤波器偏置 |
| `hyena_filter_groups` | int | 1 | 滤波分组数 |
| `fir_inner_filter_length` | int | None | 内层 FIR 滤波器长度 |
| `use_flashfft` | bool | False | FlashFFT 加速 |
| `inference_mode` | bool | True | 推理模式 |
| `column_split_hyena` | bool | True | 列拆分模式 |
| `hyena_flip_x1x2` | bool | False | 翻转 x1/x2 |
| `use_flash_depthwise` | bool | False | FlashDepthwiseConv1d |
| `depthwise_dtype` | torch.dtype | bfloat16 | 深度卷积精度 |
| `long_fir_threshold` | int | None | 长 FIR 阈值 |
| `interleave` | bool | False | 交错模式 |
| `prefill_style` | str | fft | 预填充风格 |
| `bidirectional` | bool | True | 双向滤波 |

---

## 四、小参数/小显存配置建议

### 核心策略

**1. 降低 hidden_size**
这是最直接有效的方式。hidden_size 减半，所有子模块的显存和计算量大致减少 75%。

**2. 增大 hyena_filter_groups**
增大分组数意味着每组处理的通道数减少。例如 hyena_filter_groups=hidden_size（默认值）意味着每组只有 1 通道，显存最小。如果设为 hidden_size//2，每组 2 通道，显存略增。建议保持默认值或增大此值。

**3. 降低 state_size 和 num_filters**
这两个参数控制 Hyena 滤波器的内部容量。较小的值意味着较少的可学习参数和较小的显存占用。

**4. 使用低精度 (bfloat16)**
`hyena_block_dtype=torch.bfloat16` 和 `mlp_dtype=torch.bfloat16` 确保低精度计算。

**5. 去掉所有偏置**
`qkv_proj_bias=False` 和 `hyena_out_proj_bias=False` + `short_filter_bias=False` 减少参数。

**6. 避免使用 FP8**
`use_fp8_input_projections=False`（默认），FP8 需要特殊硬件支持。

**7. 避免 torch.compile**
`compile=False`（默认），torch.compile 可能增加额外开销。

**8. 选择合理的滤波器长度**
`short_filter_length` 和 `fir_inner_filter_length` 越小，需要的显存越少，但感受野也越小。建议在可接受范围内尽量小。

### 推荐配置示例

```python
ParallelGatedConvBlock(
    hidden_size=2048,           # 原来可能是 4096，减少 60%+ 参数
    layer_idx=0,
    qkv_proj_bias=False,        # 去掉偏置
    hyena_out_proj_bias=False,  # 去掉偏置
    hyena_filter_groups=2048,   # 每组1通道，显存最小
    hyena_block_dtype=torch.bfloat16,
    mlp_dtype=torch.bfloat16,
    use_fp8_input_projections=False,
    compile=False,
    # kwargs 传递给 HyenaCascade:
    state_size=16,              # 较小值，减少参数
    num_filters=32,             # 较小值
    num_attention_heads=16,     # 减少头数
    short_filter_length=5,      # 较小的滤波器长度
    short_filter_bias=False,
    fir_inner_filter_length=None,  # 不启用内层滤波，减少复杂度
    use_flashfft=False,
    inference_mode=True,
    column_split_hyena=True,
    hyena_flip_x1x2=False,
    use_flash_depthwise=False,
    bidirectional=True,
)
```

### 参数量估算

假设原始配置为 hidden_size=4096, state_size=64, num_filters=64, short_filter_length=81：

| 配置对比 | 参数量变化 | 显存变化 | 计算量变化 |
|----------|-----------|---------|-----------|
| hidden_size 4096→2048 | ~75%↓ | ~60%↓ | ~75%↓ |
| state_size 64→16 | ~90%↓ | ~75%↓ | ~90%↓ |
| num_filters 64→32 | ~50%↓ | ~50%↓ | ~50%↓ |
| short_filter_length 81→5 | ~95%↓ | ~95%↓ | ~95%↓ |
| 无内层滤波 | 无变化 | ~20%↓ | ~20%↓ |
| bf16 vs fp32 | 无变化 | ~50%↓ | 无变化 |
| 去掉 bias | ~1%↓ | 无变化 | 无变化 |

组合以上所有策略，相比原始配置，预计可减少 **80-90%** 的参数量和显存占用。

### 注意事项

- Hyena 架构与传统 Transformer 不同，它使用递归滤波（IIR/FIR）而非注意力机制
- 缩短滤波器长度会减少感受野，可能影响模型性能
- state_size 过小可能导致数值不稳定
- 内层 FIR 滤波器（fir_inner_filter_length）提供额外表达能力，但增加显存
- Hyena 架构在长序列上通常比 Transformer 更省显存，因为其复杂度是 O(N) 而非 O(N²)
- 如果显存仍然紧张，可以考虑减小 batch_size 或 sequence_length

---

*生成时间: 2026-04-21*