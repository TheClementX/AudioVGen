import torch
import math
import dac

"""
implementation of DiT blocks Positional Encoding and MaskVat_adaln
all attention mechanisms use batch first conventions.

input dims
    TODO: switch dac time embed dims remove batch in audio encodings
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
        c -> (batch, c_dim)
        """
        conditions = self.adaln_mod(c)
        """
        1d vectors (batch, seq_len, embed_dim)
        each of these emebeddings are added token wise for temporal context
        """
        a1, a2, b1, b2, g1, g2 = conditions.chunk(6, dim=2)

        norm1 = self.layernorm1(x) * (1 + g1) + b1
        atn, _ = self.atn(norm1, norm1, norm1)
        gate1 = atn * a1

        residual1 = gate1 + x

        norm2 = self.layernorm2(residual1) * (1 + g2) + b2
        ff = self.ff(norm2)
        gate2 = ff * a2

        residual2 = gate2 + residual1

        # initially acts as an identity and returns x
        return residual2, c


class PositionalEncoding(torch.nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        pe_matrix = torch.zeros((seq_len, embed_dim))

        # (1, seq_len)
        pos = torch.arange(0, seq_len, dtype=torch.float()).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        # pos * divterm -> (seq_len, embed_dim // 2)
        # evens
        pe_matrix[:, 0::2] = torch.sin(pos * div_term)
        # odds
        pe_matrix[:, 1::2] = torch.cos(pos * div_term)
        # (batch, seq_len, embed_dim)
        pe_matrix = pe_matrix.unsqueeze(0)

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
            input = module(*input)
        return input


class MaskVatAdaLN(torch.nn.Module):
    def __init__(self, seq_len, embed_dim, n_heads, c_dim, s_dim, M, K, codebook_size):
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

        adaln = [AdaLNZero(c_dim + s_dim, embed_dim, n_heads) for _ in range(M)]
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

        return conditions

    def forward(self, d, c, s):
        """
        args: d = dac, c = clip, s = s3d
        final output: (batch, time, k, 1024)
        """

        dac_embeddings = self.get_embeddings(d)
        conditions = self.make_conditioning(c, s)

        DiT, _ = self.backbone(dac_embeddings, conditions)
        linear = self.final_linear(DiT)
        # (batch, seq_len, codebook * K) -> (batch, seq_len, K, codebook_size
        reshape = linear.reshape((-1, self.seq_len, self.K, self.codebook_size))

        return reshape

    """
    decode the DAC tensor into wav form
    """

    def decode(self, predictions):
        """
        predictions: (batch, seq_len, K)
        """
        # print(predictions.shape)
        codes = torch.permute(predictions, (0, 2, 1))

        dac_model_path = dac.utils.download(model_type="44khz")
        dac_model = dac.DAC.load(dac_model_path)
        dac_model.to("cuda")
        dac_model.eval()

        with torch.no_grad():
            z, _, _ = dac_model.quantizer.from_codes(codes)
            audio = dac_model.decode(z)

        # (batch, channels, len)
        return audio
