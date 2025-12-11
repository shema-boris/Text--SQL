'''from src.models.encoder import Encoder
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


from src.data.tokenizer import Vocabulary

if __name__ == "__main__":
    PAD = "<pad>"
    SOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"

    vocab = Vocabulary(specials=[PAD, SOS, EOS, UNK])
    sequences = [
        ["select", "name", "from", "employees"],
        ["select", "age", "from", "employees"],
        ["what", "is", "the", "average", "age"],
    ]
    vocab.build_from_sequences(sequences, min_freq=1)

    print("Vocab size:", vocab.size)
    print("Index of <pad>:", vocab.token_to_id(PAD))
    print("Index of select:", vocab.token_to_id("select"))
    print("Token for 0:", vocab.id_to_token(0))

'''
from src.data.tokenizer import Vocabulary
from src.data.dataset import TextToSQLDataset

print("\n--- Dataset test ---")
PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

src_vocab = Vocabulary([PAD, SOS, EOS, UNK])
trg_vocab = Vocabulary([PAD, SOS, EOS, UNK])

# For this quick test, build vocabs from a few toy sentences
src_sequences = [
    ["what", "is", "the", "average", "age", "?"],
    ["how", "many", "employees", "are", "there", "?"],
]
trg_sequences = [
    ["select", "avg", "(", "age", ")", "from", "people"],
    ["select", "count", "(", "*", ")", "from", "employees"],
]

src_vocab.build_from_sequences(src_sequences, min_freq=1)
trg_vocab.build_from_sequences(trg_sequences, min_freq=1)

dataset = TextToSQLDataset(
    path="data/raw/toy_train.jsonl",
    src_vocab=src_vocab,
    trg_vocab=trg_vocab,
    max_src_len=16,
    max_trg_len=16,
)

print("Dataset size:", len(dataset))
example = dataset[0]
print("src_ids:", example["src_ids"])
print("trg_ids:", example["trg_ids"])
print("src_len:", example["src_len"])
print("trg_len:", example["trg_len"])
