import torch
from src.data.tokenizer import Vocabulary
from src.data.dataset import TextToSQLDataset
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

def build_vocabs_from_dataset(path: str, max_src_len: int, max_trg_len: int) -> tuple[Vocabulary, Vocabulary]:
    src_vocab = Vocabulary([PAD, SOS, EOS, UNK])
    trg_vocab = Vocabulary([PAD, SOS, EOS, UNK])

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



@torch.no_grad()
def translate_question(
    model: Seq2Seq,
    src_vocab: Vocabulary,
    trg_vocab: Vocabulary,
    question: str,
    device: torch.device,
) -> str:
    model.eval()

    # 1) Tokenize question like in TextToSQLDataset.tokenize_question
    tokens = question.lower().strip().split()

    # 2) Convert to ids
    src_ids = [src_vocab.token_to_id(tok) for tok in tokens]
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, src_len)
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long, device=device)      # (1,)

    # 3) Run model in inference mode (trg=None -> greedy decoding)
    logits = model(src=src_tensor, src_lengths=src_lengths, trg=None, teacher_forcing_ratio=0.0)
    # logits: (1, out_len, output_dim)

    # 4) Get predicted ids by argmax at each step
    preds = logits.argmax(dim=-1)[0]  # (out_len,)

    # 5) Convert ids to tokens, stop at <eos>
    sql_tokens: list[str] = []
    eos_idx = trg_vocab.token_to_id(EOS)
    pad_idx = trg_vocab.token_to_id(PAD)

    for idx in preds.tolist():
        if idx == eos_idx:
            break
        if idx == pad_idx:
            continue
        tok = trg_vocab.id_to_token(idx)
        # Skip <sos> if it appears
        if tok in {SOS, EOS, PAD}:
            continue
        sql_tokens.append(tok)

    # Simple join; you can later improve spacing around punctuation
    sql = " ".join(sql_tokens)
    return sql


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = "data/raw/train.jsonl"
    max_src_len = 64
    max_trg_len = 64

    # Rebuild vocabs from training data
    src_vocab, trg_vocab = build_vocabs_from_dataset(train_path, max_src_len, max_trg_len)

    # Build a fresh model with same architecture as during training
    model = build_model(src_vocab, trg_vocab, device)

    # Load trained weights
    checkpoint_path = "checkpoints/text2sql.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Enter a question (or 'quit' to exit):")
    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        sql = translate_question(model, src_vocab, trg_vocab, question, device)
        print("Predicted SQL:")
        print(sql)


if __name__ == "__main__":
    main()