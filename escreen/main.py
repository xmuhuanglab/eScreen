import os,subprocess
from . import genome_tool
from . import miaomiao_tool
from .motif_tool import load_pwm_from_meme_c,load_pwm_from_meme_c1,CenteredMaxPool1D

from .vortex.main.ParallelGatedConvBlock import ParallelGatedConvBlock
from .vortex.main.AttentionBlock import AttentionBlock
from .vortex.main.RMSNorm import RMSNorm
from .vortex.main.HyenaInferenceEngine import fftconv_standard

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from einops import rearrange
import sys,copy,json


########################
# escreen architecture
########################
class eScreen(nn.Module):
    def __init__(self,kernel_fwd,kernel_rev=None,d_model=32,num_filters=32,seq_length=500,celltype_num=32,mode='sequence',device='cpu',lr=1e-5):
        super().__init__()
        # 定义模型架构
        self.kernel_fwd = kernel_fwd # 正向motif,不可学习
        self.kernel_rev = kernel_rev # 逆向motif,不可学习
        self.pwm_thresh = nn.Parameter(torch.tensor([0.4])) # 认为是motif的最小阈值,可学习
        
        self.d_model = d_model # 特征维度数
        self.seq_length = seq_length # 最长序列长度
        self.celltype_num = celltype_num # 细胞类型数
        
        self.HyenaCluster=nn.ModuleList([ # Hyena模块簇
            self.get_block(
                d_model=d_model,layer_idx=0,num_filters=d_model,num_attention_heads=8,hyena_filter_groups=128,fir_inner_filter_length=7,block_type='HyenaSE'
            ),
            self.get_block(
                d_model=d_model,layer_idx=1,num_filters=d_model,num_attention_heads=8,hyena_filter_groups=128,fir_inner_filter_length=128,block_type='HyenaMR'
            ),
            self.get_block(
                d_model=d_model,layer_idx=2,num_filters=d_model,num_attention_heads=8,block_type='HyenaLI'
            ),
            self.get_block(
                d_model=d_model,layer_idx=3,num_attention_heads=8,block_type='Attention'
            )
        ])
        
        self.motif_embedding    = nn.Linear(self.kernel_fwd.size(0),d_model) # 编码motif的embeding
        self.celltype_embedding = nn.Embedding(num_embeddings=self.celltype_num, embedding_dim=d_model) # 编码celltype的embeding
        self.CenterMaxPool = CenteredMaxPool1D(kernel_size=19, stride=1)
        
        self.header=nn.Sequential(
            nn.Linear(d_model*(seq_length+celltype_num),d_model),
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model,d_model),
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model,1),
        )

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
                              mha_out_proj_bias  = True,
                              qkv_proj_bias      = False,
                              use_flash_attn     = False,      # 因为cuda的问题(不支持CUDA 12.4),这里不能用flash attention
                              inner_size_multiple_of = 16,     # 这五个参数是MLP的参数
                              mlp_activation         = 'gelu', # 这五个参数是MLP的参数
                              evo2_style_activations = True,   # 这五个参数是MLP的参数
                              model_parallel_size    = 1,      # 这五个参数是MLP的参数
                              inner_mlp_size         = 256     # 这五个参数是MLP的参数
            ).to(torch.bfloat16)
            
    def move(self):
        self.kernel_fwd = self.kernel_fwd.to(self.device)
        self.kernel_rev = self.kernel_rev.to(self.device)
        self.pwm_thresh.data = self.pwm_thresh.data.to(self.device)
        self.motif_embedding.to(self.device)
        self.celltype_embedding.to(self.device)
        self.CenterMaxPool.to(self.device)
        self.header.to(self.device)
        for layer in self.HyenaCluster:
            layer.to(self.device)
        
    def forward(self,x,z):
        
        ### 扫motif,这个时候的形状从原始的(b,l,4)变为(b,4,l)再变为(b, ck, l)      
        x = rearrange( x, 'b l c -> b c l')
        if not self.kernel_rev is None:
            x = fftconv_standard(signal = x,kernel = self.kernel_fwd,bias=None,padding=17,device = self.device) +\
                fftconv_standard(signal = x,kernel = self.kernel_rev,bias=None,padding=17,device = self.device)
        else:
            x = fftconv_standard(signal = x,kernel = self.kernel_fwd,bias=None,padding=17,device = self.device)
        x = nn.functional.relu(x-self.pwm_thresh)
        x = self.CenterMaxPool(x)
        x = rearrange( x, 'b c l -> b l c') # 把通道放到最后一个维度
        x = self.motif_embedding(x)         # 把特征映射到d_model
        z = self.celltype_embedding(z)      # 生成维度为d_model的软嵌入
        
        x  = torch.concatenate([x,z],dim=1) # 把motif嵌入和软嵌入合在一起,形状变为(batch,ls+lc,d_model)

        
        ### 通过stripedHyena层和header层,形状维持(batch,ls+lc,d_model)
        x = x.to(torch.bfloat16)
        for layer in self.HyenaCluster:
            x = layer(x)
            x = nn.functional.relu(x)
            
        ### 最后一步展平嵌入进行打分 
        x = x.to(torch.float).view(x.size(0),-1)
        x = self.header(x).flatten()
        x = nn.functional.sigmoid(x)    
        return torch.clip(x,0,1)
    
    def fit(self,train_loader,valid_loader=None,epochs=200,lr=1e-5,check_step=100,earlystop=20,device='cpu',save_name='./torch_logs/best_model'):
        
        optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.device = device
        best_valid_acc = 0.0; count = 0; best_model = None; valid_count = 0
        batch_size = train_loader.batch_size
        for epoch in range(epochs):
            epoch_loss = 0
            for i, batch in enumerate(train_loader):
                x,y,z = batch;x = x.to(device);y = y.to(device);z = z.to(device)
                self.move();self.train()
                optimizer.zero_grad()

                p = self(x,z)

                loss = F.binary_cross_entropy(p,y)
                #loss = F.mse_loss(p,y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                avg_loss = epoch_loss / (i+1)
                print(f"Step/Epoch [{(i+1)*batch_size}/{epoch+1}], Loss: {avg_loss:.4f}",end='\r')
                
                valid_count += 1
                if valid_count >= check_step:
                    if not valid_loader is None:
                        p,y = self.predict(valid_loader,device=device,verbose=True,with_true=True)
                        valid_acc    = spearmanr(p,y)[0]
                        if valid_acc > best_valid_acc:
                            best_valid_acc = valid_acc
                            count          = 0
                            torch.save(self.state_dict(),save_name+'.best.pt')
                            print(f"Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Val-acc: {valid_acc:.4f} ↑",end='\n')
                        else:
                            count += 1
                            print(f"Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Val-acc: {valid_acc:.4f} -",end='\n')
                            if count > earlystop:
                                print(f'Model early stop in Epoch {epoch+1} with valid Acc {best_valid_acc:.4f}')
                                self.load_state_dict(torch.load(save_name+'.best.pt'))
                                break
                        valid_count = 0
                    else:
                        print("")
                        valid_count = 0
        torch.save(self.state_dict(),save_name+'.final.pt')
        return None
    
    def predict(self,data_loader,device='cpu',verbose=True,with_true=False):
        y_pred=[];y_true=[];self.eval();self.device=device;self.move()
        with torch.no_grad():
            if verbose:
                for batch in tqdm(data_loader, leave=True):
                    if with_true:
                        x,y,z = batch;x = x.to(device);y = y.to(device);z = z.to(device)
                        preds = self.forward(x,z)
                        y_pred.extend(preds.cpu().numpy())
                        y_true.extend(y.cpu().numpy())
                    else:
                        x,y,z = batch;x = x.to(device);z = z.to(device)
                        preds = self.forward(x,z)
                        y_pred.extend(preds.cpu().numpy())
            else:
                for batch in data_loader:
                    if with_true:
                        x,y,z = batch;x = x.to(device);y = y.to(device);z = z.to(device)
                        preds = self.forward(x,z)
                        y_pred.extend(preds.cpu().numpy())
                        y_true.extend(y.cpu().numpy())
                    else:
                        x,y,z = batch;x = x.to(device);z = z.to(device)
                        preds = self.forward(x,z)
                        y_pred.extend(preds.cpu().numpy()) 
                    
        return np.array(y_pred),np.array(y_true)