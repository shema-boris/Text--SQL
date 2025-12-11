import torch
from torch import nn
from typing import Tuple

class Encoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int, num_layers: int=1, dropout: float=0.1, bidirectional: bool=True) -> None:
        super().__init__()
        #Map token ids -> dense vectors
        self.embedding=nn.Embedding(num_embeddings=input_dim, embedding_dim=embed_dim, padding_idx=0)
        self.lstm=nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.bidirectional=bidirectional
    def forward(self,src:torch.Tensor, src_lengths: torch.Tensor)->Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # src -> (batch_size, sr_seq_len, embed_dim)
        embedded=self.embedding(src)
        outputs, (hidden,cell)=self.lstm(embedded)

        return outputs, (hidden, cell)
    