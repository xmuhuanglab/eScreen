import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class CellEncoder(nn.Module):
    def __init__(self, num_genes, d_model=256, nhead=4, num_layers=4):
        super().__init__()

        # gene identity
        self.gene_emb = nn.Embedding(num_genes, d_model)

        # expression encoder
        self.expr_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # transformer encoder（无 positional encoding！）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # attention pooling
        self.attn_pool = nn.Linear(d_model, 1)

    def forward(self, gene_ids, exprs, log1p = True):
        """
        gene_ids: (B, N)
        exprs:    (B, N)
        """

        # gene embedding
        g_emb = self.gene_emb(gene_ids)  # (B, N, d)

        # expression embedding
        if not log1p: # 如果输入不是log1p
            exprs = torch.log1p(exprs)  # 很关键！
        e_emb = self.expr_mlp(exprs.unsqueeze(-1))  # (B, N, d)

        # gating（比简单相加强）
        x = g_emb * torch.sigmoid(e_emb)

        # transformer
        x = self.transformer(x)

        # attention pooling
        attn = torch.softmax(self.attn_pool(x), dim=1)  # (B, N, 1)
        cell_emb = (attn * x).sum(dim=1)  # (B, d)

        return cell_emb, x  # x用于mask任务
    
def mask_expression(exprs, mask_ratio=0.15):
    mask = torch.rand(exprs.shape, device=exprs.device) < mask_ratio
    masked_exprs = exprs.clone()
    masked_exprs[mask] = 0.0
    return masked_exprs, mask

class ExprDecoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)  # (B, N)
    
def augment(exprs):
    # dropout genes
    drop_mask = torch.rand(exprs.shape, device=exprs.device) < 0.1
    exprs = exprs.clone()
    exprs[drop_mask] = 0

    # 加噪声
    noise = torch.randn_like(exprs) * 0.1
    return exprs + noise


class CellEmb(nn.Module):
    def __init__(self, num_genes=19215, d_model=256):
        super().__init__()
        self.encoder = CellEncoder(num_genes=num_genes)
        self.decoder = ExprDecoder(d_model=d_model)
    
    def forward(self, gene_ids, exprs, mask_ratio=0.15):
        """
        完整的模型前向传播
        Args:
            gene_ids: (B, N) gene indices
            exprs: (B, N) expression values
            mask_ratio: ratio of genes to mask for training
        Returns:
            pred_expr: (B, N) predicted expressions
            cell_emb: (B, d_model) cell embedding
        """
        # Masked modeling forward
        masked_exprs, mask = mask_expression(exprs, mask_ratio)
        cell_emb, token_emb = self.encoder(gene_ids, masked_exprs)
        pred_expr = self.decoder(token_emb)
        
        return pred_expr, cell_emb, mask
    
    def encode_cell(self, gene_ids, exprs):
        """只编码，不解码，用于获取细胞表征"""
        cell_emb, _ = self.encoder(gene_ids, exprs)
        return cell_emb
    
    def decode(self, token_emb):
        """只解码"""
        return self.decoder(token_emb)
    

    
    
##################################################################
## 也可以用GeneFormer生成embedding
##################################################################
import pickle
import pandas as pd
import numpy as np
from transformers import BertForMaskedLM

class Geneformer(nn.Module):
    
    def __init__(self, TOKEN_DICTIONARY_FILE, GENE_MEDIAN_FILE, ENSEMBL_DICTIONARY_FILE, 
                 ENSEMBL_MAPPING_FILE, MODEL_PATH, PREDIFINED_EMB=None, model_version="V1",device="cpu"):

        # 加载模型
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
        
        # V1 vs V2 配置
        if self.model_version == "V1":
            self.MAX_LENGTH = 2048
            self.SPECIAL_TOKEN = False
        elif self.model_version == "V2":
            self.MAX_LENGTH = 4096
            self.SPECIAL_TOKEN = True
        else:
            raise ValueError(f"model_version must be 'V1' or 'V2', got {model_version}")
        
        # 加载基因->Token字典
        with open(self.TOKEN_DICTIONARY_FILE, 'rb') as f:
            self.gene_token_dict = pickle.load(f)
        self.token_gene_dict = {v: k for k, v in self.gene_token_dict.items()}
        
        # 获取特殊 token IDs (可能不存在于 V1)
        self.cls_token_id = self.gene_token_dict.get('<cls>')
        self.eos_token_id = self.gene_token_dict.get('<eos>')
        self.pad_token_id = self.gene_token_dict.get('<pad>', 0)
        
        # 加载基因中位数（用于归一化）
        if self.GENE_MEDIAN_FILE:
            with open(self.GENE_MEDIAN_FILE, 'rb') as f:
                self.gene_median_dict = pickle.load(f)
        
        # 加载基因Symbol映射到Ensembl ID的字典        
        if self.ENSEMBL_DICTIONARY_FILE:
            with open(self.ENSEMBL_DICTIONARY_FILE, 'rb') as f:
                self.gene_symbol_to_ensembl = pickle.load(f)    
        
        # 加载Ensembl ID映射字典 (用于版本号清理和ID规范化)
        if self.ENSEMBL_MAPPING_FILE:
            with open(self.ENSEMBL_MAPPING_FILE, 'rb') as f:
                self.ensembl_mapping_dict = pickle.load(f)
        else:
            # 如果没有提供，使用恒等映射
            self.ensembl_mapping_dict = {k: k for k in self.gene_token_dict.keys()}
        
        self.model.eval()
        self.device=device
        
        if not PREDIFINED_EMB is None:
            self.PREDIFINED_EMB = PREDIFINED_EMB.to(self.device)

    def _tokenize_cell(self, expression_list, gene_list, gene_type='DepMap', target_sum=10000):
        """
        将单个细胞的表达数据转换为 token IDs
        遵循 Geneformer TranscriptomeTokenizer 的原始逻辑

        Args:
            gene_list: list of str, 基因名列表
            expression_list: list of float, 对应的表达量列表
            gene_type: str, 'DepMap' (Gene Symbol) 或 'Ensembl'
            target_sum: 归一化目标总和 (默认 10000)

        Returns:
            input_ids: list of token IDs, 如果无有效基因返回 None
        """
        # 1. 过滤未检测到的基因（表达量 > 0）
        # 确保两个列表长度相同
        if len(gene_list) != len(expression_list):
            raise ValueError("gene_list and expression_list must have the same length")

        # 过滤表达量 > 0 的基因
        detected_genes = []
        detected_exprs = []
        for gene, expr in zip(gene_list, expression_list):
            if expr > 0:
                detected_genes.append(gene)
                detected_exprs.append(expr)

        if len(detected_genes) == 0:
            return None

        # 2. 整合基因并获取 token 和表达量
        gene_expr_dict = {}  # {ensembl_id: 表达量}

        for gene, expr in zip(detected_genes, detected_exprs):
            ensembl_id = None

            # 基因名映射到 Ensembl ID
            if gene_type == 'DepMap':
                # DepMap 格式: "TP53 10" 或 "TP53"
                gene_symbol = gene.split(' ')[0]
                if gene_symbol in self.gene_symbol_to_ensembl:
                    ensembl_id = self.gene_symbol_to_ensembl[gene_symbol]
                else:
                    continue

            elif gene_type == 'Ensembl':
                # Ensembl 格式: "ENSG00000141510" 或 "ENSG00000141510.11"
                raw_ensembl_id = gene.split('.')[0] if '.' in gene else gene
                ensembl_id = raw_ensembl_id

            if ensembl_id is None:
                continue

            # 步骤1: 通过 ensembl_mapping_dict 映射/清理 Ensembl ID
            if ensembl_id in self.ensembl_mapping_dict:
                ensembl_id = self.ensembl_mapping_dict[ensembl_id]

            # 步骤2: 整合相同 ensembl_id 的表达量（求和）
            if ensembl_id in gene_expr_dict:
                gene_expr_dict[ensembl_id] += expr
            else:
                gene_expr_dict[ensembl_id] = expr

        if len(gene_expr_dict) == 0:
            return None

        # 步骤3: 过滤出有 token 的基因
        valid_ensembl_ids = [eid for eid in gene_expr_dict.keys() if eid in self.gene_token_dict]
        if len(valid_ensembl_ids) == 0:
            return None

        # 4. 准备归一化所需的数组
        gene_tokens = []
        expression_values = []
        norm_factors = []

        for ensembl_id in valid_ensembl_ids:
            gene_tokens.append(self.gene_token_dict[ensembl_id])
            expression_values.append(gene_expr_dict[ensembl_id])
            norm_factors.append(self.gene_median_dict.get(ensembl_id, 1.0))

        # 5. 归一化表达量
        total_counts = sum(expression_values)
        if total_counts == 0:
            return None

        expression_array = np.array(expression_values, dtype=np.float32)
        norm_array = np.array(norm_factors, dtype=np.float32)
        norm_expr = (expression_array / total_counts * target_sum) / norm_array

        # 6. 按归一化后的表达量降序排序
        sorted_indices = np.argsort(-norm_expr)
        sorted_tokens = np.array(gene_tokens)[sorted_indices]

        # 7. 截断到最大长度
        if self.SPECIAL_TOKEN:
            max_seq_len = self.MAX_LENGTH - 2
        else:
            max_seq_len = self.MAX_LENGTH
        sorted_tokens = sorted_tokens[:max_seq_len]

        # 8. 添加特殊 token
        if self.SPECIAL_TOKEN:
            if self.cls_token_id is None or self.eos_token_id is None:
                raise ValueError("CLS or EOS token ID is None. Check token dictionary.")
            input_ids = [self.cls_token_id] + sorted_tokens.tolist() + [self.eos_token_id]
        else:
            input_ids = sorted_tokens.tolist()

        return input_ids

    def _extract_embedding(self, input_ids, emb_type='cell', emb_layer=-1, device='cuda'):
        """
        提取 embeddings，遵循 Geneformer EmbExtractor 的原始逻辑

        Args:
            input_ids: list of token IDs
            emb_type: str, 可选 'cell', 'gene', 'cls'
                      - 'cell': 对所有 gene token 的 embedding 进行 mean pooling
                      - 'cls': CLS token embedding (仅 V2)
                      - 'gene': 返回每个 token 对应的 embedding
            emb_layer: int, -1:倒数第二层, 0:最后一层
            device: str, 设备

        Returns:
            emb_type='cell': torch.tensor, shape [hidden_dim]
            emb_type='cls': torch.tensor, shape [hidden_dim]
            emb_type='gene': dict, {token_id: embedding_tensor}
        """
        if input_ids is None:
            return None
        
        original_len = len(input_ids)
        
        # 准备输入：padding 到 MAX_LENGTH
        if original_len < self.MAX_LENGTH:
            pad_len = self.MAX_LENGTH - original_len
            input_ids_padded = input_ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * original_len + [0] * pad_len
        else:
            input_ids_padded = input_ids[:self.MAX_LENGTH]
            attention_mask = [1] * self.MAX_LENGTH
        
        input_tensor = torch.tensor([input_ids_padded]).to(device)
        attention_mask_tensor = torch.tensor([attention_mask]).to(device)

        # 前向传播
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor, 
                attention_mask=attention_mask_tensor,
                output_hidden_states=True
            )

        # 计算层索引 (Geneformer 原始逻辑)
        total_layers = len(outputs.hidden_states)
        if emb_layer == -1:
            # 倒数第二层 (2nd to last)
            layer_idx = total_layers - 2
        elif emb_layer == 0:
            # 最后一层 (last)
            layer_idx = total_layers - 1
        else:
            layer_idx = emb_layer

        embeddings = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_dim]
        original_lens = torch.tensor([original_len], device=device)

        # 根据 emb_type 返回 (遵循 Geneformer 原始逻辑)
        if emb_type == 'cls':
            if not self.SPECIAL_TOKEN:
                raise ValueError("CLS token not available when SPECIAL_TOKEN=False (V1 model)")
            # CLS token 在位置 0
            return embeddings[0, 0, :].cpu()

        elif emb_type == 'cell':
            # Cell embedding: mean pooling of gene embeddings
            if self.SPECIAL_TOKEN:
                # V2: 排除 CLS (pos 0) 和 EOS (pos -1)
                # 使用原始长度 - 2 作为有效长度
                valid_len = original_len - 2
                if valid_len <= 0:
                    return None
                gene_embs = embeddings[0, 1:original_len-1, :]
            else:
                # V1: 所有 token 都是 gene tokens
                valid_len = original_len
                gene_embs = embeddings[0, :original_len, :]
            
            # 只取有效部分进行 mean pooling
            embedding = gene_embs[:valid_len, :].mean(dim=0)
            return embedding.cpu()

        elif emb_type == 'gene':
            # Gene embedding: 返回每个 token 对应的 embedding
            gene_embeddings = {}
            # 只返回实际序列中的 token (不包含 padding)
            for i, token_id in enumerate(input_ids):
                gene_embeddings[token_id] = embeddings[0, i, :].cpu()
            return gene_embeddings

        else:
            raise ValueError(f"emb_type must be 'cell', 'gene', or 'cls', got {emb_type}")
            
    def get_embedding(self, x, z, gene_type='DepMap', target_sum=10000, 
                      emb_type='cell', emb_layer=-1, device='cuda'):
        """
        获取细胞或基因的 embedding
        
        Args:
            x: list, 表达量
            z: list, 基因名称列表
            gene_type: 'DepMap' (Gene Symbol) 或 'Ensembl'
            target_sum: 归一化目标总和
            emb_type: 'cell', 'gene', 'cls'
            emb_layer: -1 (倒数第二层) 或 0 (最后一层)
            device: 计算设备
        
        Returns:
            相应的 embedding
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
        x是输入,可以是token化之后的细胞系，也可以是[(表达值,基因列表),...]
        """
        if torch.max(x) < len(self.PREDIFINED_EMB) and isinstance(x,torch.Tensor): 
            # 如果细胞系在预先定义的embedding中并且已经转换成embedding
            # PREDIFINED_EMB的形状是[Predifined_cell_type_num,D]
            return self.PREDIFINED_EMB[x]
        else:
            emb = []
            for _x in x:
                emb.append(
                    get_embedding(self, _x[0], _x[1], gene_type=gene_type, target_sum=10000, emb_type=emb_type, emb_layer=emb_layer, device=self.device)
                )
            return torch.stack(emb)
        
        
        
            