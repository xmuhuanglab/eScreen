import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import numpy as np
import os
os.environ['TE']='0'

from utils import boost_sampler

from SH2core.ParallelGatedConvBlock import ParallelGatedConvBlock
from SH2core.AttentionBlock import AttentionBlock
from SH2core.RMSNorm import RMSNorm
from SH2core.HyenaInferenceEngine import fftconv_standard

from einops import rearrange
from tqdm import tqdm

class eSCREEN_backbone(nn.Module):
    def __init__(self,filter_type='conv',d_model=32,num_filters=32,seq_length=500,celltype_num=32,mode='sequence',device='cpu',lr=1e-5,**kwargs):
        super().__init__()
        # 定义模型架构
        if filter_type == 'conv': # 使用随机初始化的卷积层进行嵌入
            padding = 2
            self.conv0 = nn.Sequential(
                nn.Conv1d(in_channels=4, out_channels=d_model//2, kernel_size=19, padding='same'),
                nn.ReLU(),
                nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2, kernel_size=11, padding='same'),
                nn.ReLU(),
                nn.Conv1d(in_channels=d_model//2, out_channels=d_model, kernel_size=7, padding='same'),
            ).to(torch.bfloat16)
            self.motif_embedding = nn.Linear(d_model,d_model).to(torch.bfloat16) # 编码motif的embeding
        elif filter_type == 'pwm': # 使用Motif PWM进行嵌入
            kernel_fwd = kwargs.get('kernel_fwd')
            kernel_rev = kwargs.get('kernel_fwd')
            kernel_size = kernel_fwd.shape[-1]
            # 计算确定的padding值：padding = (kernel_size - 1) // 2
            padding = (kernel_size - 1) // 2
            self.conv0 = nn.Conv1d(in_channels=4, out_channels=2*len(kernel_fwd), kernel_size=kernel_fwd.shape[-1], padding='same')
            # 假设 kernel_fwd 和 kernel_rev 的形状为 [num_motifs, 4, kernel_size]
            # 将它们拼接在维度0上（输出通道维度）
            combined_weight = torch.cat([kernel_fwd, kernel_rev], dim=0)
            self.conv0.weight.data = combined_weight
            self.conv0.weight.requires_grad = False
            self.conv0 = self.conv0.to(torch.bfloat16)
            self.motif_embedding = nn.Linear(2*len(kernel_fwd),d_model).to(torch.bfloat16) # 编码motif的embeding
        elif filter_type == 'pwm-conv': # PWM混合卷积
            pass # 暂时不支持，不考虑
        elif filter_type == 'emb': # 使用embedding层
            self.emb_layer = nn.Embedding(num_embeddings=6, embedding_dim=d_model, padding_idx=0)
            
        self.filter_type = filter_type
        
        self.d_model = d_model # 特征维度数
        self.seq_length = seq_length # 最长序列长度
        
        self.HyenaCluster=nn.ModuleList([ # Hyena模块簇
            self.get_block(
                d_model=d_model,layer_idx=0,num_filters=d_model,num_attention_heads=8,hyena_filter_groups=128,fir_inner_filter_length=7,block_type='HyenaSE',**kwargs
            ),
            self.get_block(
                d_model=d_model,layer_idx=1,num_filters=d_model,num_attention_heads=8,hyena_filter_groups=128,fir_inner_filter_length=128,block_type='HyenaMR',**kwargs
            ),
            self.get_block(
                d_model=d_model,layer_idx=2,num_filters=d_model,num_attention_heads=8,block_type='HyenaLI',**kwargs
            ),
            self.get_block(
                d_model=d_model,layer_idx=3,num_attention_heads=8,block_type='Attention',**kwargs
            ),
        ])

        self.device=device
        
    def get_block(self,block_type='HyenaLI',**kwargs):
        if block_type=='HyenaSE':
            return ParallelGatedConvBlock(
                              hidden_size        = kwargs['d_model'],
                              layer_idx          = kwargs['layer_idx'],
                              qkv_proj_bias      = False,
                              hyena_out_proj_bias= True,   
                              state_size  = 16,
                              num_filters = kwargs['num_filters'], # 注意这个参数不能比hidden_size大
                              num_attention_heads= kwargs['num_attention_heads'], # 这个参数和后面的attention层一致即可
                              short_filter_length= 3,
                              short_filter_bias  = False,
                              hyena_filter_groups    = kwargs['hyena_filter_groups'],     # 区分HyenaMR和HyenaSE,不能比d_model更大,需要被d_model整除
                              fir_inner_filter_length= kwargs['fir_inner_filter_length'], # 参数是区分HyenaMR和HyenaSE的关键
                              inner_size_multiple_of = 16,      # 这五个参数是MLP的参数
                              mlp_activation         = 'gelu',  # 这五个参数是MLP的参数
                              evo2_style_activations = True,    # 这五个参数是MLP的参数
                              model_parallel_size    = 1,       # 这五个参数是MLP的参数
                              inner_mlp_size         = 256,     # 这五个参数是MLP的参数
                              column_split_hyena     = False, # 这个是HyenaCascade的参数
                              interleave             = True,  # 这个是HyenaCascade的参数
                              hyena_flip_x1x2        = False, # 这个是HyenaCascade的参数
                              use_flash_depthwise    = False, # 这个是HyenaCascade的参数
                              use_flashfft           = False, # 这个是HyenaCascade的参数
                              prefill_style          = 'fft', # 这个是HyenaCascade的参数
                              bidirectional          = True   # 这个是HyenaCascade的参数
            ).to(torch.bfloat16)
        elif block_type=='HyenaMR':
            return ParallelGatedConvBlock(
                              hidden_size= kwargs['d_model'],
                              layer_idx  = kwargs['layer_idx'],
                              qkv_proj_bias=False,
                              hyena_out_proj_bias=True,
                              state_size = 16,
                              num_filters= kwargs['num_filters'],
                              num_attention_heads    = kwargs['num_attention_heads'], # 这个参数和后面的attention层一致即可
                              short_filter_length    = 3,
                              short_filter_bias      = False,
                              hyena_filter_groups    = kwargs['hyena_filter_groups'],     # 区分HyenaMR和HyenaSE,不能比d_model更大,需要被d_model整除
                              fir_inner_filter_length= kwargs['fir_inner_filter_length'], # 参数是区分HyenaMR和HyenaSE的关键
                              inner_size_multiple_of = 16,     # 这五个参数是MLP的参数
                              mlp_activation         = 'gelu', # 这五个参数是MLP的参数
                              evo2_style_activations = True,   # 这五个参数是MLP的参数
                              model_parallel_size    = 1,      # 这五个参数是MLP的参数
                              inner_mlp_size         = 256,    # 这五个参数是MLP的参数
                              column_split_hyena     = False, # 这个是HyenaCascade的参数
                              interleave             = True,  # 这个是HyenaCascade的参数
                              hyena_flip_x1x2        = False, # 这个是HyenaCascade的参数
                              use_flash_depthwise    = False, # 这个是HyenaCascade的参数
                              use_flashfft           = False, # 这个是HyenaCascade的参数
                              prefill_style          = 'fft', # 这个是HyenaCascade的参数
                              bidirectional          = True   # 这个是HyenaCascade的参数
            ).to(torch.bfloat16)
        elif block_type=='HyenaLI':
            return ParallelGatedConvBlock(
                              hidden_size= kwargs['d_model'],
                              layer_idx  = kwargs['layer_idx'],
                              qkv_proj_bias=False,
                              hyena_out_proj_bias=True,
                              state_size = 16,
                              num_filters= kwargs['num_filters'],
                              num_attention_heads= kwargs['num_attention_heads'], # 这个参数和后面的attention层一致即可
                              short_filter_length= 3,
                              short_filter_bias  = False,
                              proj_groups        = 1,
                              # 没有hyena_filter_groups和fir_inner_filter_length两个参数时就变为HyenaLI
                              # 原来的配置里use_flashfft=False,所以这里也不传递快速卷积模块
                              inner_size_multiple_of =16,     # 这五个参数是MLP的参数
                              mlp_activation         ='gelu', # 这五个参数是MLP的参数
                              evo2_style_activations =True,   # 这五个参数是MLP的参数
                              model_parallel_size    =1,      # 这五个参数是MLP的参数
                              inner_mlp_size         =256,    # 这五个参数是MLP的参数
                              column_split_hyena     = False, # 这个是HyenaCascade的参数
                              interleave             = True,  # 这个是HyenaCascade的参数
                              hyena_flip_x1x2        = False, # 这个是HyenaCascade的参数
                              use_flash_depthwise    = False, # 这个是HyenaCascade的参数
                              use_flashfft           = False, # 这个是HyenaCascade的参数
                              prefill_style          = 'fft', # 这个是HyenaCascade的参数
                              bidirectional          = True   # 这个是HyenaCascade的参数
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
                              use_flash_attn     = kwargs.get('use_flash_attn',False),  # CUDA 12.4才能用flash attention
                              inner_size_multiple_of = 16,     # 这五个参数是MLP的参数
                              mlp_activation         = 'gelu', # 这五个参数是MLP的参数
                              evo2_style_activations = True,   # 这五个参数是MLP的参数
                              model_parallel_size    = 1,      # 这五个参数是MLP的参数
                              inner_mlp_size         = 256     # 这五个参数是MLP的参数
            ).to(torch.bfloat16)
            
    def move(self):
        self.conv0.to(self.device)
        self.motif_embedding.to(self.device)
        self.CenterMaxPool.to(self.device)
        self.header.to(self.device)
        for layer in self.HyenaCluster:
            layer.to(self.device)
        
    def p_count(self):
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
        
        if self.filter_type == 'emb':
            ### 直接嵌入,这个时候的形状从原始的(b,l)变为(b,l,d)
            x = self.emb_layer(x.to(torch.int))
        else:
            ### 扫motif,这个时候的形状从原始的(b,l,4)变为(b,4,l)再变为(b, ck, l)
            x = rearrange( x, 'b l c -> b c l').to(torch.bfloat16)
            x = nn.functional.relu(self.conv0(x))
            x = rearrange( x, 'b c l -> b l c') # 把通道放到最后一个维度
            x = self.motif_embedding(x)         # 把特征映射到d_model
        
        ### 通过stripedHyena层和header层,形状维持(batch,ls+lc,d_model)
        x = x.to(torch.bfloat16)
        for layer in self.HyenaCluster:
            x = x + layer(x)

        emb = x.mean(dim=1)
        return emb.float()
    

    
class FlexibleFFN(nn.Module):
    """灵活的前馈网络，支持可变层数"""
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.1):
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
        return self.net(x)
    
    

class CellConditionedLoRA(nn.Module):
    """
    Cell-Conditioned LoRA: 为每个细胞系生成低秩变换矩阵
    delta = x @ A @ B
    其中 A = f_A(ct_emb) ∈ [B, D, r]
         B = f_B(ct_emb) ∈ [B, r, D]
    
    初始化策略：B初始化为零 → delta初始为0 → 初始时等价于恒等映射
    """
    def __init__(self, d_model, rank=4, alpha=1.0):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        
        # 生成A矩阵（down-projection）
        self.f_A = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, d_model * rank)
        )
        
        # 生成B矩阵（up-projection）
        self.f_B = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, rank * d_model)
        )
        
        # 关键初始化：f_B的最后一层权重和偏置置零
        # 这样初始时B=0，delta=0，不改变原始表示
        nn.init.zeros_(self.f_B[-1].weight)
        nn.init.zeros_(self.f_B[-1].bias)
        
        # A正常初始化（小值）
        nn.init.normal_(self.f_A[-1].weight, std=0.02)
        nn.init.zeros_(self.f_A[-1].bias)
        
        # 可学习的缩放因子，初始化为小值
        self.alpha = nn.Parameter(torch.tensor(alpha))
    
    def forward(self, x, ct_emb):
        """
        x: [B, L, D] 或 [B, D]（兼容pooled特征）
        ct_emb: [B, D]
        """
        B = x.size(0)
        
        # 生成A和B
        A_flat = self.f_A(ct_emb)  # [B, D * rank]
        B_flat = self.f_B(ct_emb)  # [B, rank * D]
        
        A = A_flat.view(B, self.d_model, self.rank)  # [B, D, r]
        B = B_flat.view(B, self.rank, self.d_model)  # [B, r, D]
        
        # 计算 delta = x @ A @ B
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
    def __init__(self, d_model, K=8):
        super().__init__()
        self.K = K
        # K 个原型头，每个都是 d_model -> 1 的线性层（或小MLP）
        self.proto_W = nn.Parameter(torch.randn(K, d_model, 1) * 0.02)
        self.proto_b = nn.Parameter(torch.zeros(K, 1))
        # 路由器：细胞嵌入 -> 注意力分数
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, K)
        )

    def forward(self, h, ct_emb):
        # h: [B, d_model]
        attn_scores = self.router(ct_emb)          # [B, K]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, K]
        # 每个原型的输出：[B, K, 1]
        # 批量计算：h [B,1,d_model]  @ proto_W [K,d_model,1] -> [B,K,1]
        proto_out = torch.einsum('bd,kdo->bko', h, self.proto_W).squeeze(-1) + self.proto_b.squeeze(-1)  # [B, K]
        out = (proto_out * attn_weights).sum(dim=-1, keepdim=True)  # [B, 1]
        #return torch.sigmoid(out)
        return out
    

class eScreen_vX(nn.Module):
    """eScreen_vX with aux BCE head (identical to M2a version)."""
    def __init__(self, backbone, d_model, cell_emb, d_cell_emb=256,
                 cell_lora_rank=64, MoE_K=64,
                 freeze_backbone=False, freeze_cellemb=False,
                 freeze_celllora=False, freeze_header=False):
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
        if optimizer is None:
            optimizer = AdamW(self.parameters(), lr=3e-4, weight_decay=0.01)

        best_valid_acc = float('-inf')
        count = 0
        valid_count = 0

        use_boost = False
        sampler_mode = str(sampler).lower().strip()
        if sampler_mode == 'none':
            pass
        elif sampler_mode in ('label_boost', 'label', 'boost') or use_boost:
            use_boost = True
        else:
            raise ValueError(f"Unknown sampler: {sampler!r}")

        def _make_booster():
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
