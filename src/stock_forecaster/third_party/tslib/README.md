# Vendored TSLib (Time-Series-Library)

Source: [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)

Files adapted for the `stock_forecaster` package (imports only; model logic unchanged):

- `layers/embed.py` — `DataEmbedding_inverted`
- `layers/transformer_enc_dec.py` — `Encoder`, `EncoderLayer`
- `layers/self_attention.py` — `FullAttention`, `AttentionLayer`
- `utils/masking.py` — `TriangularCausalMask`
- `models/itransformer.py` — `Model` (iTransformer)

Paper: [iTransformer (arXiv:2310.06625)](https://arxiv.org/abs/2310.06625)
