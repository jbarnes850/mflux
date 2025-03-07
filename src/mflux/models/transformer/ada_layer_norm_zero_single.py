import mlx.core as mx
from mlx import nn


class AdaLayerNormZeroSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3072, 3 * 3072)
        self.norm = nn.LayerNorm(dims=3072, eps=1e-6, affine=False)

    def __call__(self, x: mx.array, text_embeddings: mx.array):
        text_embeddings = self.linear(nn.silu(text_embeddings))
        chunk_size = 9216 // 3
        shift_msa = text_embeddings[:, 0 * chunk_size : 1 * chunk_size]
        scale_msa = text_embeddings[:, 1 * chunk_size : 2 * chunk_size]
        gate_msa = text_embeddings[:, 2 * chunk_size : 3 * chunk_size]
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa
