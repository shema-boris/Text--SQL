import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.tokenizer import Vocabulary
from src.data.dataset import TextToSQLDataset
from src.data.collate import text_to_sql_collate_fn
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq


PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"


def build_vocabs_from_dataset(path: str, max_src_len: int, max_trg_len: int) -> tuple[Vocabulary, Vocabulary]:
    """
    Very simple vocab builder:
    - Loads dataset once to collect token sequences.
    - Builds src and trg vocabs.
    """
    src_vocab = Vocabulary([PAD, SOS, EOS, UNK])
    trg_vocab = Vocabulary([PAD, SOS, EOS, UNK])

    # Temporary dataset to get raw tokens
    tmp_dataset = TextToSQLDataset(
        path=path,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        max_src_len=max_src_len,
        max_trg_len=max_trg_len,
    )

    src_sequences = [ex.src_tokens for ex in tmp_dataset.examples]
    trg_sequences = [ex.trg_tokens for ex in tmp_dataset.examples]

    src_vocab.build_from_sequences(src_sequences, min_freq=1)
    trg_vocab.build_from_sequences(trg_sequences, min_freq=1)

    return src_vocab, trg_vocab


def build_model(src_vocab: Vocabulary, trg_vocab: Vocabulary, device: torch.device) -> Seq2Seq:
    pad_idx = src_vocab.token_to_id(PAD)
    sos_idx = trg_vocab.token_to_id(SOS)
    eos_idx = trg_vocab.token_to_id(EOS)

    input_dim = src_vocab.size
    output_dim = trg_vocab.size

    embed_dim = 128
    enc_hidden_dim = 256
    dec_hidden_dim = 256
    num_layers = 1
    dropout = 0.1

    encoder = Encoder(
        input_dim=input_dim,
        embed_dim=embed_dim,
        hidden_dim=enc_hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
    )

    # If encoder is bidirectional, enc_hidden_dim_out = enc_hidden_dim * 2
    enc_hidden_dim_out = enc_hidden_dim * 2

    decoder = Decoder(
        output_dim=output_dim,
        embed_dim=embed_dim,
        enc_hidden_dim=enc_hidden_dim_out,
        dec_hidden_dim=dec_hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        max_output_len=64,
    )

    return model.to(device)


def train_one_epoch(
    model: Seq2Seq,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float = 0.5,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in data_loader:
        src = batch["src_batch"].to(device)
        trg = batch["trg_batch"].to(device)
        src_lengths = batch["src_lengths"].to(device)

        # Shift target for input and output
        trg_input = trg[:, :-1]
        trg_target = trg[:, 1:]

        optimizer.zero_grad()

        logits = model(
            src=src,
            src_lengths=src_lengths,
            trg=trg_input,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        # logits: (batch_size, out_len, output_dim)

        batch_size, out_len, output_dim = logits.size()
        logits_flat = logits.reshape(batch_size * out_len, output_dim)
        targets_flat = trg_target.reshape(batch_size * out_len)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * (batch_size * out_len)
        # Count non-pad tokens for averaging
        non_pad = (targets_flat != criterion.ignore_index).sum().item()
        total_tokens += max(non_pad, 1)

    return total_loss / max(total_tokens, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = "data/raw/train.jsonl"
    max_src_len = 64
    max_trg_len = 64
    batch_size = 32
    lr = 1e-3
    num_epochs = 5
    teacher_forcing_ratio = 0.5

    # 1) Build vocabularies from training data
    src_vocab, trg_vocab = build_vocabs_from_dataset(train_path, max_src_len, max_trg_len)

    pad_idx = src_vocab.token_to_id(PAD)

    # 2) Build train dataset and dataloader
    train_dataset = TextToSQLDataset(
        path=train_path,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        max_src_len=max_src_len,
        max_trg_len=max_trg_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: text_to_sql_collate_fn(batch, pad_idx=pad_idx),
    )

    # 3) Build model, optimizer, loss
    model = build_model(src_vocab, trg_vocab, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 4) Training loop
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        print(f"Epoch {epoch}: avg loss per token = {avg_loss:.4f}")


if __name__ == "__main__":
    main()