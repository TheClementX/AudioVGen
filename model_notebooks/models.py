import torch

class AdaLNZero(torch.nn.Module): 
    def __init__(self, d_cond, d_model, n_heads): 
        super().__init__()

        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.atn = torch.nn.MultiheadAttention(d_model, n_heads)

        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, 2 * d_model), 
            torch.nn.Linear(2 * d_model, d_model)
        )

        self.adaln_mod = torch.nn.Sequential(
            #activation
            torch.nn.SiLU(), 
            torch.nn.Linear(d_cond, 6 * d_model)
        )

        torch.nn.init.constant_(self.adaln_mod[-1].weight, 0)
        torch.nn.init.constant_(self.adaln_mod[-1].bias, 0)

    def forward(self, x, c):
        """
        x -> (batch, seq_len, embed_dim)
        c -> (batch, c_dim)
        """
        conditions = self.adaln_mod(c)
        #1d vectors
        a1, a2, b1, b2, g1, g2 = conditions.chunk(6, dim=1)

        norm1 = self.layernorm1(x) * (1 + g1.unsqueeze(0)) + b1.unsqueeze(0)
        atn = self.atn(norm1, norm1, norm1)
        gate1 = atn * (1 + a1)

        residual1 = gate1 + x

        norm2 = self.layernorm2(residual1) * (1 + g2.unsqueeze(0)) + b2.unsqueeze(0)
        ff = self.ff(norm2)
        gate2 = ff * (1 + a2)

        residual2 = gate2 + residual1 

        return residual2

class PositionalEncoding(torch.nn.Module):
    def __init__(self): 
        super().__init__()
        pass

    def forward(self): 
        pass


# class MaskVatAdaLN(torch.nn.Module): 
#     def __init__(self): 
#         super().__init__()
#         pass

#     def forward(self): 
#         pass