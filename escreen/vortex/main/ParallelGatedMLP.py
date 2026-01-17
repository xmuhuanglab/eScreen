import torch.nn as nn
import torch.nn.functional as F
from typing import Optional





def grab_first_if_tuple(x):
    if x.__class__.__name__ == "tuple":
        return x[0]
    else:
        return x

class ParallelGatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        inner_size_multiple_of: int = 64,
        mlp_activation: str = "gelu",
        evo2_style_activations: bool = False,
        model_parallel_size: int = 1,
        inner_mlp_size: Optional[int] = None,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError(f"Activation function {self.act_type} not implemented")
        
        if self.layer_idx > 0 and evo2_style_activations:
            self.act = nn.Identity()

        self.multiple_of = multiple_of * model_parallel_size

        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * ((inner_size + self.multiple_of - 1) // self.multiple_of)
        inner_size = inner_mlp_size if inner_mlp_size is not None else inner_size

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        z1, z2 = grab_first_if_tuple(z1), grab_first_if_tuple(z2)
        y = self.l3(self.act(z1) * z2)
        return grab_first_if_tuple(y)