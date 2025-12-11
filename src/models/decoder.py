import torch
from torch import nn
from typing import Tuple

from .attention import Attention


class Decoder(nn.Module):
    """
    Decoder LSTM with attention for generating SQL tokens.
    """

    def __init__(
        self,
        output_dim: int,
        embed_dim: int,
        enc_hidden_dim: int,
        dec_hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Embedding for target (SQL) tokens
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=embed_dim,
            padding_idx=0,  # will match target <pad> index later
        )

        # Attention module
        self.attention = Attention(
            enc_hidden_dim=enc_hidden_dim,
            dec_hidden_dim=dec_hidden_dim,
        )

        # LSTM that takes [embedded_token ; context_vector] as input
        self.lstm = nn.LSTM(
            input_size=embed_dim + enc_hidden_dim,
            hidden_size=dec_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Final projection: from [decoder_hidden ; context] to vocab logits
        self.fc_out = nn.Linear(dec_hidden_dim + enc_hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.output_dim = output_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.num_layers = num_layers

    def forward(
        self,
        input_token: torch.Tensor,                        # (batch_size,)
        prev_hidden: Tuple[torch.Tensor, torch.Tensor],   # (h, c)
        encoder_outputs: torch.Tensor,                    # (batch_size, src_seq_len, enc_hidden_dim)
        mask: torch.Tensor | None = None,                 # (batch_size, src_seq_len)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        One decoding step.

        Returns:
            output_logits: (batch_size, output_dim)
            hidden: (h, c), each (num_layers, batch_size, dec_hidden_dim)
        """
        h, c = prev_hidden  # each: (num_layers, batch_size, dec_hidden_dim)

        # For attention we usually use the top layer hidden state at current time
        # Take last layer: h[-1] -> (batch_size, dec_hidden_dim)
        dec_hidden_for_attn = h[-1]

        # 1) Compute context vector using attention
        #    context: (batch_size, enc_hidden_dim)
        context = self.attention(dec_hidden_for_attn, encoder_outputs, mask)

        # 2) Embed current input token
        #    input_token: (batch_size,) -> (batch_size, 1, embed_dim)
        embedded = self.embedding(input_token).unsqueeze(1)
        embedded = self.dropout(embedded)

        # 3) Concatenate embedded token and context for LSTM input
        #    context:   (batch_size, enc_hidden_dim) -> (batch_size, 1, enc_hidden_dim)
        #    rnn_input:(batch_size, 1, embed_dim + enc_hidden_dim)
        context_expanded = context.unsqueeze(1)
        rnn_input = torch.cat([embedded, context_expanded], dim=-1)

        # 4) LSTM step
        #    outputs: (batch_size, 1, dec_hidden_dim)
        #    hidden, cell: each (num_layers, batch_size, dec_hidden_dim)
        outputs, (h_new, c_new) = self.lstm(rnn_input, (h, c))

        # Take the output at this step: (batch_size, dec_hidden_dim)
        dec_output = outputs.squeeze(1)

        # 5) Compute logits over vocabulary using [decoder_output ; context]
        #    concat: (batch_size, dec_hidden_dim + enc_hidden_dim)
        concat = torch.cat([dec_output, context], dim=-1)
        logits = self.fc_out(concat)  # (batch_size, output_dim)

        return logits, (h_new, c_new)