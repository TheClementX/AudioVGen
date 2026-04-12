import torch
import math
import dac
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba2

"""
implementation of DiT blocks Positional Encoding and MaskVat_adaln
all attention mechanisms use batch first conventions.

input dims
    -> DAC : (seq_len, embed_dim)
    -> BEATs : (seq_len, embed_dim) -> 
    -> CLIP : (seq_len, embed_dim) -> (seq_len, 512)
    -> S3D : (seq_len, embed_dim) -> (seq_len, 1024)
"""

class AdaLNZero(torch.nn.Module):
    def __init__(self, d_cond, d_model, n_heads):
        super().__init__()

        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.atn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, 2 * d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * d_model, d_model),
        )

        self.adaln_mod = torch.nn.Sequential(
            # activation
            torch.nn.SiLU(),
            torch.nn.Linear(d_cond, 6 * d_model),
        )

        torch.nn.init.constant_(self.adaln_mod[-1].weight, 0)
        torch.nn.init.constant_(self.adaln_mod[-1].bias, 0)

    def forward(self, x, c):
        """
        x -> (batch, seq_len, embed_dim)
        c -> (batch, seq_len, c_dim)
        """
        conditions = self.adaln_mod(c)
        """
        1d vectors (batch, seq_len, embed_dim)
        each of these emebeddings are added token wise for temporal context
        These values are initially 0 so this block acts as an identity
        """
        a1, a2, b1, b2, g1, g2 = conditions.chunk(6, dim=2)

        norm1 = self.layernorm1(x) * (1 + g1) + b1
        atn, _ = self.atn(norm1, norm1, norm1, need_weights=False)
        gate1 = atn * a1

        residual1 = gate1 + x

        norm2 = self.layernorm2(residual1) * (1 + g2) + b2
        ff = self.ff(norm2)
        gate2 = ff * a2

        residual2 = gate2 + residual1

        # initially acts as an identity and returns x
        return residual2, c


class BiMamba2(torch.nn.Module): 
    """
    A regular BiMamba block using mamab 2 architecture
    https://arxiv.org/pdf/2404.15772 : bidirectional mamba
    """
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2): 
        super().__init__()
        self.m_forward = Mamba2(
            d_model = d_model, 
            d_state = d_state, 
            expand = expand, 
            d_conv = d_conv
        )

        self.m_backward = Mamba2(
            d_model = d_model, 
            d_state = d_state, 
            expand = expand, 
            d_conv = d_conv
        )

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, 2 * d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * d_model, d_model),
        )

        self.norm_forward = torch.nn.LayerNorm(d_model)
        self.norm_backward = torch.nn.LayerNorm(d_model)
        self.norm_final = torch.nn.LayerNorm(d_model)

    def forward(self, x): 
        """
        x : (batch, seq_len, embed_dim)
        """

        #forward mamba
        y_forward = self.m_forward(x)
        y_forward = self.norm_forward(y_forward + x)

        #backward mamba
        x_flip = torch.flip(x, dims=[1])
        y_backward = self.m_backward(x_flip)
        y_backward = torch.flip(y_backward, dims=[1])
        y_backward = self.norm_backward(y_backward+x)

        #combine forward and backward
        y = y_forward + y_backward

        #feed forward
        y_ff = self.ff(y)

        #final output
        out = self.norm_final(y_ff + y)

        return out

class BiMamba2AdaLN2(torch.nn.Module): 
    """
    a variation of mamba adaln where adaln is only used before bimamba and 
    feed forward
    """

    def __init__(self, d_model, d_cond, d_state=64, d_conv=4, expand=2): 
        super().__init__()
        #main blocks
        self.mamba = BiMamba2(d_model, d_state, d_conv, expand)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 2), 
            torch.nn.ReLU(), 
            torch.nn.Linear(d_model * 2, d_model)
        )

        #layer norms
        self.ln_mamba = torch.nn.LayerNorm(d_model)
        self.ln_ff = torch.nn.LayerNorm(d_model)

        #adaln modulation
        self.adaln_mod = torch.nn.Sequential(
            torch.nn.SiLU(), 
            torch.nn.Linear(d_cond, 6 * d_model)
        )

    def forward(self, x, c): 
        conditions = self.adaln_mod(c)
        a1, b1, g1, a2, b2, g2 = conditions.chunk(6, dim=2)

        norm1 = self.ln_mamba(x) * (1 + g1) + b1
        mamba_out = self.mamba(norm1)
        gate1 = mamba_out * a1

        residual =  x + gate1

        norm2 = self.ln_ff(residual)
        ff_out = self.ff(norm2)
        gate2 = ff_out * a1

        out = gate2 + residual

        return out, c

class BiMamba2AdaLN3(torch.nn.Module): 
    """
    A novel AdaLN bi-mamba block architecutre
    https://arxiv.org/pdf/2404.15772 : bidirectional mamba
    """
    def __init__(self, d_model, d_cond, d_state=64, d_conv=4, expand=2): 
        super().__init__()
        self.m_forward = Mamba2(
            d_model = d_model, 
            d_state = d_state, 
            expand = expand, 
            d_conv = d_conv
        )

        self.m_backward = Mamba2(
            d_model = d_model, 
            d_state = d_state, 
            expand = expand, 
            d_conv = d_conv
        )

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, 2 * d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * d_model, d_model),
        )

        #adaln modulation
        self.adaln_mod = torch.nn.Sequential(
            # activation
            torch.nn.SiLU(),
            torch.nn.Linear(d_cond, 9 * d_model),
        )

        torch.nn.init.constant_(self.adaln_mod[-1].weight, 0)
        torch.nn.init.constant_(self.adaln_mod[-1].bias, 0)

        self.norm_forward = torch.nn.LayerNorm(d_model)
        self.norm_backward = torch.nn.LayerNorm(d_model)
        self.norm_final = torch.nn.LayerNorm(d_model)

    def forward(self, x, c): 
        """
        x : (batch, seq_len, embed_dim)
        """
        #create conditions
        conditions = self.adaln_mod(c)
        a1, a2, a3, b1, b2, b3, g1, g2, g3 = conditions.chunk(9, dim=2)
        #flip for reversed mamba block
        g2 = torch.flip(g2, dims=[1])
        b2 = torch.flip(b2, dims=[1])

        #forward mamba
        norm_f = self.norm_forward(x) * (1 + g1) + b1
        x_f = self.m_forward(norm_f)
        gate_f = a1 * x_f

        #backward
        x_flip = torch.flip(x, dims=[1])
        norm_b = self.norm_backward(x_flip) * (1 + g2) + b2
        x_b = self.m_backward(norm_b)
        x_b = torch.flip(x_b, dims=[1])
        gate_b = a2 * x_b

        #combined residual
        residual = gate_b + gate_f + x

        #combine
        norm_out = self.norm_final(residual) * (1 + g3) + b3
        ff_out = self.ff(norm_out)

        gate3 = ff_out * a3
        out = gate3 + residual

        return out, c

class PositionalEncoding(torch.nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        pe_matrix = torch.zeros((seq_len, embed_dim))

        # (1, seq_len)
        pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        # pos * divterm -> (seq_len, embed_dim // 2)
        # evens
        pe_matrix[:, 0::2] = torch.sin(pos * div_term)
        # odds
        pe_matrix[:, 1::2] = torch.cos(pos * div_term)
        # (batch, seq_len, embed_dim)
        pe_matrix = pe_matrix.unsqueeze(0).to(torch.bfloat16)

        self.register_buffer("pe_matrix", pe_matrix)

    def forward(self, x):
        """
        x -> (batch, seq_len, embed_dim)
        """
        seq_len = x.shape[1]
        return x + self.pe_matrix[:, :seq_len, :]

class MultiSequential(torch.nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if self.training: 
                input = checkpoint(module, *input,  use_reentrant=False)
            else: 
                input = module(*input)
        return input

class AdaLNMamba2(torch.nn.Module): 
    def __init__(self, d_model, d_cond, num_heads, d_state, d_conv, expand, ratio=4): 
        super().__init__()
        self.dit = AdaLNZero(d_cond, d_model, num_heads)
        
        mamba = [BiMamba2AdaLN2(d_model, d_cond, d_state, d_conv, expand) for _ in range(ratio)]
        self.mamba = MultiSequential(*mamba)

    def forward(self, x, c): 
        x, c = self.dit(x, c)
        y, c = self.mamba(x, c)
        return y, c

class AudioVGen(torch.nn.Module):
    def __init__(
            self, seq_len, embed_dim, n_heads, 
            d_state, d_conv, expand, 
            c_dim, s_dim, 
            M, K, codebook_size, ratio=4
        ):
        """
        mask vectors are the 1025 vector in each layer
        args:
            seq_len : DAC sequence length
            embed_dim : model embedding dimension
            n_heads : number of heads for multihead attention
            c_dim : embedding dimension size of CLIP
            s_dim : embedding dimension size for S3D
            M : number of AdaLNZero blocks
            K : DAC codebook depth in layers
            codebook_size : number of tokens in DAC codebook
        result:
            dac_in : (batch, seq_len, K)
            preds_out : (batch, seq_len, K, codebook_size)
        """
        super().__init__()

        # model parameters
        self.embed_dim = embed_dim
        self.K = K
        self.seq_len = seq_len
        self.codebook_size = codebook_size

        num_embeddings = codebook_size * K + K
        # (batch, 1, K)
        embed_offset = torch.tensor([[codebook_size * i for i in range(K)]]).unsqueeze(
            0
        )
        self.register_buffer("embed_offset", embed_offset)

        self.embedding = torch.nn.Embedding(num_embeddings, embed_dim)
        self.pose_dac = PositionalEncoding(self.seq_len, self.embed_dim)
        self.pose_conditions = PositionalEncoding(self.seq_len, c_dim + s_dim)

        # clip mlp
        self.c_mlp = torch.nn.Sequential(
            torch.nn.Linear(c_dim, 2 * c_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * c_dim, c_dim),
        )

        # s3d mlp
        self.s_mlp = torch.nn.Sequential(
            torch.nn.Linear(s_dim, 2 * s_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * s_dim, s_dim),
        )

        adaln = [AdaLNMamba2(embed_dim, c_dim + s_dim, n_heads, d_state, d_conv, expand, ratio=ratio) for _ in range(M)]
        self.backbone = MultiSequential(*adaln)

        # (batch, seq_len, embed_dim) -> (batch, seq_len, codebook * K)
        self.final_linear = torch.nn.Linear(embed_dim, codebook_size * K)

    def get_embeddings(self, d):
        """
        d : dac encodings
            -> (batch, seq_len, code_layers) -> (N, L, K)

        returns:
        e -> (batch, seq_len, embed_dim)
        """
        # print(torch.max(d))
        # print(torch.max(self.embed_offset))
        offset_embeds = d + self.embed_offset
        # print(torch.max(offset_embeds))
        e = self.embedding(offset_embeds)
        # print(torch.max(e))
        e = torch.sum(e, dim=2)
        e = self.pose_dac(e)

        return e

    def make_conditioning(self, c, s):
        """
        c : clip encodings
            -> (batch, seq_len, c_dim)
        s : s3d encodings
            -> (batch, seq_len, s_dim)
        """
        # (batch, embed_dim, seq_len)
        mlp_out_c = self.c_mlp(c).permute(0, 2, 1)
        mlp_out_s = self.s_mlp(s).permute(0, 2, 1)

        # (batch, seq_len, embed_dim)
        interp_c = torch.nn.functional.interpolate(
            mlp_out_c, size=self.seq_len, mode="nearest-exact"
        ).permute(0, 2, 1)
        interp_s = torch.nn.functional.interpolate(
            mlp_out_s, size=self.seq_len, mode="nearest-exact"
        ).permute(0, 2, 1)

        # (batch, seq_len, 2 * embed_dim)
        conditions = torch.concat([interp_c, interp_s], dim=2)
        conditions = self.pose_conditions(conditions)

        return conditions

    def forward(self, d, c, s):
        """
        args: d = dac, c = clip, s = s3d
        final output: (batch, time, k, 1024)
        """

        dac_embeddings = self.get_embeddings(d)
        conditions = self.make_conditioning(c, s)

        if self.training:
            batch = conditions.shape[0]
            drop_mask = (torch.rand(batch, 1, 1, device=conditions.device) > 0.1).to(conditions.dtype)
            conditions = conditions * drop_mask

        DiT, _ = self.backbone(dac_embeddings, conditions)
        linear = self.final_linear(DiT)
        # (batch, seq_len, codebook * K) -> (batch, seq_len, K, codebook_size
        reshape = linear.reshape((-1, self.seq_len, self.K, self.codebook_size))

        return reshape

    """
    decode the DAC tensor into wav form
    """

    def decode(self, model, predictions, chunk_size=8):
        """
        predictions: (batch, seq_len, K)
        return (raw audio) : (batch, channels, len)
        process in chhunk_size sized mini batches 
        """
        # (batch, K, seq_len)
        codes = torch.permute(predictions, (0, 2, 1))
        batch, K, seq_len = codes.shape

        decoded_audio = []

        #process a full batch in mini batches 
        with torch.no_grad():
            for i in range(0, batch, chunk_size): 
                batch_codes = codes[i:i+chunk_size]
                z, _, _ = model.quantizer.from_codes(batch_codes)
                audio = model.decode(z)
                decoded_audio.append(audio)

        audio = torch.cat(decoded_audio, dim=0)

        # (batch, channels, len)
        return audio