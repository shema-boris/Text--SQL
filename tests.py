from src.models.encoder import Encoder
from src.models.attention import Attention
import torch

encoder = Encoder(
    input_dim=1000,
    embed_dim=128,
    hidden_dim=256,
    num_layers=1,
    bidirectional=True,
)

src = torch.randint(0, 1000, (2, 5))  # (batch_size=2, seq_len=5)
lengths = torch.tensor([5, 3], dtype=torch.long)

outputs, (h, c) = encoder(src, lengths)
print(outputs.shape)  # expect: (2, 5, 512)  # 256 * 2 directions
print(h.shape)        # expect: (2, 2, 256)  # (num_layers * num_directions, batch, hidden



batch_size = 2
src_seq_len = 5
enc_hidden_dim = 8
dec_hidden_dim = 8

attn = Attention(enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim)

decoder_hidden = torch.randn(batch_size, dec_hidden_dim)
encoder_outputs = torch.randn(batch_size, src_seq_len, enc_hidden_dim)
mask = torch.tensor([[1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1]], dtype=torch.long)

context = attn(decoder_hidden, encoder_outputs, mask)
print(context.shape)  # expect: (2, 8)