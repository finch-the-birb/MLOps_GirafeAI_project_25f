"""iTransformer backbone (vendored from TSLib with multimodal extensions)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from stock_forecaster.third_party.tslib.layers.embed import DataEmbeddingInverted
from stock_forecaster.third_party.tslib.layers.multimodal_embed import MultimodalITransformerEmbedding
from stock_forecaster.third_party.tslib.layers.self_attention import AttentionLayer, FullAttention
from stock_forecaster.third_party.tslib.layers.transformer_enc_dec import Encoder, EncoderLayer


class Model(nn.Module):
    """TSLib iTransformer with optional multimodal inverted embedding."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.multimodal_mode = getattr(configs, "multimodal_mode", None)
        self.n_numeric_features = getattr(configs, "n_numeric_features", configs.enc_in)
        self.text_feat_dim = getattr(configs, "text_feat_dim", 32)

        if self.multimodal_mode is not None:
            self.enc_embedding = MultimodalITransformerEmbedding(
                seq_len=configs.seq_len,
                d_model=configs.d_model,
                n_numeric_features=self.n_numeric_features,
                text_feat_dim=self.text_feat_dim,
                multimodal_mode=self.multimodal_mode,
                dropout=configs.dropout,
                n_heads=configs.n_heads,
                text_m_dim=getattr(configs, "text_m_dim", 1),
            )
        else:
            self.enc_embedding = DataEmbeddingInverted(
                configs.seq_len,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        if self.task_name == "classification":
            self.act = functional.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def encode_multimodal(self, x_numeric: torch.Tensor, x_text: torch.Tensor) -> torch.Tensor:
        enc_out = self.enc_embedding(x_numeric, x_text)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        if self.multimodal_mode == "flatten":
            enc_out = enc_out[:, : self.n_numeric_features, :]
        return enc_out

    def classification(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None) -> torch.Tensor:
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        del x_dec, x_mark_dec, mask
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        return None
