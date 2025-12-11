import torch
from torch import nn
from typing import Optional

from .encoder import Encoder
from .decoder import Decoder


class Seq2Seq(nn.Module):
    """
    Wrapper around Encoder and Decoder for training and inference.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        sos_idx: int,
        eos_idx: int,
        pad_idx: int,
        max_output_len: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.max_output_len = max_output_len

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: (batch_size, src_seq_len)
        Returns:
            mask: (batch_size, src_seq_len), 1 for real tokens, 0 for PAD
        """
        return (src != self.pad_idx).long()

    def forward(
        self,
        src: torch.Tensor,                      # (batch_size, src_seq_len)
        src_lengths: torch.Tensor,              # (batch_size,)
        trg: Optional[torch.Tensor] = None,     # (batch_size, trg_seq_len) during training
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        If trg is provided: training mode, return logits for each target step.
        If trg is None: inference mode (greedy), generate up to max_output_len.

        Returns:
            logits: (batch_size, out_seq_len, output_dim)
        """
        batch_size = src.size(0)

        # 1) Encode source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        # encoder_outputs: (batch_size, src_seq_len, enc_hidden_dim)

        # Build mask for attention
        src_mask = self.make_src_mask(src)  # (batch_size, src_seq_len)

        # 2) Decide output length
        if trg is not None:
            # Training: we know the full target length
            trg_len = trg.size(1)
            out_seq_len = trg_len
        else:
            # Inference: we limit by max_output_len
            trg_len = self.max_output_len
            out_seq_len = trg_len

        output_dim = self.decoder.output_dim
        # Prepare tensor for all time-step logits
        logits = torch.zeros(batch_size, out_seq_len, output_dim, device=src.device)

        # 3) Initial decoder input: <sos>
        input_token = torch.full(
            (batch_size,),
            fill_value=self.sos_idx,
            dtype=torch.long,
            device=src.device,
        )

        # Initial hidden state is encoder final state
        hidden_state = (hidden, cell)

        for t in range(out_seq_len):
            # One decoding step
            step_logits, hidden_state = self.decoder(
                input_token=input_token,
                prev_hidden=hidden_state,
                encoder_outputs=encoder_outputs,
                mask=src_mask,
            )
            # step_logits: (batch_size, output_dim)
            logits[:, t, :] = step_logits

            if trg is not None:
                # Training mode: teacher forcing
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher_forcing:
                    # Next input is ground-truth token at time t
                    input_token = trg[:, t]
                else:
                    # Next input is model's prediction at time t
                    input_token = step_logits.argmax(dim=-1)
            else:
                # Inference mode: always feed back prediction
                input_token = step_logits.argmax(dim=-1)

        return logits