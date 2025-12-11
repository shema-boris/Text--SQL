import torch
from torch import nn

class Attention(nn.Module):

    def __init__(self, enc_hidden_dim:int, dec_hidden_dim:int) -> None:
        super().__init__()
        #We will use "general" Luong attention
        # score (h_t, h_s)=h_t^T*W*h_s
        self.attn=nn.Linear(dec_hidden_dim, enc_hidden_dim, bias=False)

        #Softmax will tuen scores over time steps into probabilities
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self,decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        """
        Returns:
            context: (batch_size, enc_hidden_dim)
        """
        # 1) Project decoder hidden to encoder hidden dimension
        #    decoder_hidden: (batch_size, dec_hidden_dim)
        #    -> projected:  (batch_size, enc_hidden_dim)
        projected=self.attn(decoder_hidden)

        # 2) Compute attention scores by dot product with encoder outputs
        #    encoder_outputs: (batch_size, src_seq_len, enc_hidden_dim)
        #    projected:      (batch_size, enc_hidden_dim) -> (batch_size, 1, enc_hidden_dim)
        #    scores:         (batch_size, 1, src_seq_len)

        projected=projected.unsqueeze(1) # (batch_size, 1, enc_hidden_dim)
        scores= torch.bmm(projected, encoder_outputs.transpose(1,2))

        # 3) Apply mask (if provided) so PAD positions get -inf score
        if mask is not None:
            # mask: (batch_size, src_seq_len) -> (batch_size, 1, src_seq_len)
            mask=mask.unsqueeze(1)
            scores=scores.masked_fill(mask==0, float("-inf"))

        # 4) Softmax over src_seq_len to get attention weights
        #    attn_weights: (batch_size, 1, src_seq_len)
        attn_weights= self.softmax(scores)
        # 5) Compute context vector as weighted sum of encoder_outputs
        #    attn_weights:   (batch_size, 1, src_seq_len)
        #    encoder_outputs:(batch_size, src_seq_len, enc_hidden_dim)
        #    context:        (batch_size, 1, enc_hidden_dim)
        context = torch.bmm(attn_weights, encoder_outputs)
        #Remove the time dimension: (batch_size, enc_hidden_dim)
        context=context.squeeze(1)
        
        return context
