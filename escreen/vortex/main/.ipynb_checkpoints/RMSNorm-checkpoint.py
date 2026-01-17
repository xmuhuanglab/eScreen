import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size,eps=1e-6,use_flash_rmsnorm=False):
        # 需要的参数为:残差eps,隐藏状态数,是否使用快速归一化
        super().__init__()
        self.eps, self.hidden_size = eps, hidden_size
        self.scale = torch.nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("scale", self.scale)
        self.use_flash_rmsnorm = use_flash_rmsnorm

        if self.use_flash_rmsnorm:
            from flash_attn.ops.rms_norm import rms_norm as rmsnorm_func

            self.rmsnorm_func = rmsnorm_func

    def forward(self, x):
        if self.use_flash_rmsnorm:
            #print('PASS')
            return self.rmsnorm_func(x, self.scale, self.eps)
        else:
            #print('PASS')
            y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2) + self.eps)
            #print(x.device,y.device,self.scale.device)
            return self.scale * y