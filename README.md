# Text-to-SQL with Seq2Seq LSTM + Attention

This project is an educational implementation of a **Text-to-SQL** system:
Given a natural language question, the model generates a SQL query.

The core model is a **Seq2Seq Encoder–Decoder with Luong Attention** built in
PyTorch and trained on a simplified version of the **WikiSQL** dataset.

The main goals of this project are:

- To learn how Seq2Seq models with attention work end-to-end.
- To implement the full pipeline:
  - data loading and tokenization
  - encoder/decoder/attention modules
  - training loop with teacher forcing
  - evaluation and inference helpers
- To observe the limitations of a classic LSTM approach on a real dataset.

> This is a learning prototype, not a production-grade Text-to-SQL system.

---

## Project Structure

```text
Text--SQL/
├─ data/
│  └─ raw/
│     ├─ train.jsonl             # small toy dataset for sanity checks
│     ├─ wikisql_train.jsonl     # simplified WikiSQL train set (question + sql)
│     ├─ wikisql_dev.jsonl       # simplified WikiSQL dev set  (question + sql)
├─ checkpoints/
│  └─ text2sql.pt                # saved model weights (after training)
├─ scripts/
│  └─ prepare_wikisql.py         # convert raw WikiSQL to JSONL (question/sql)
├─ src/
│  ├─ data/
│  │  ├─ tokenizer.py            # Vocabulary class and token <-> id mapping
│  │  ├─ dataset.py              # TextToSQLDataset (reads JSONL)
│  │  └─ collate.py              # collate_fn for padding batches
│  ├─ models/
│  │  ├─ encoder.py              # BiLSTM encoder
│  │  ├─ attention.py            # Luong general attention
│  │  ├─ decoder.py              # LSTM decoder with attention
│  │  └─ seq2seq.py              # Seq2Seq wrapper (teacher forcing + decoding)
│  ├─ training/
│  │  ├─ trainer.py              # training loop (train + dev evaluation)
│  │  ├─ inference.py            # interactive inference / translate_question
│  │  └─ metrics.py              # (simple metrics helpers, if used)
│  └─ utils/
│     └─ ...                     # placeholders for helpers/logging/config
├─ tests.py                      # quick shape/tests for model components
└─ requirements.txt
```

---

## Model Overview

- **Encoder** (`src/models/encoder.py`)
  - Embedding layer
  - Bidirectional LSTM
  - Produces encoder outputs and hidden/cell states.

- **Attention** (`src/models/attention.py`)
  - Luong general attention:
    - scores decoder hidden state against encoder outputs
    - returns attention weights and context vector.

- **Decoder** (`src/models/decoder.py`)
  - LSTM with attention at each step.
  - Takes previous token + context vector, outputs next token distribution.

- **Seq2Seq wrapper** (`src/models/seq2seq.py`)
  - Orchestrates encoder forward pass.
  - Initializes decoder state from biLSTM hidden state.
  - Runs decoding loop with teacher forcing during training.
  - Uses greedy decoding at inference time.

---

## Data

Training data is stored as JSONL (`.jsonl`) files with one example per line:

```json
{"question": "What school did player number 21 play for?", "sql": "SELECT School/Club Team FROM table WHERE No. = 21"}
```

Main files:

- `data/raw/train.jsonl`  
  Small toy dataset for sanity checks and quick experiments.

- `data/raw/wikisql_train.jsonl`  
  Simplified WikiSQL training set (question + SQL string).

- `data/raw/wikisql_dev.jsonl`  
  Simplified WikiSQL dev set for validation.

`scripts/prepare_wikisql.py` contains the logic used to convert the raw WikiSQL
format into this simplified JSONL format.

---

## Setup

```bash
# clone this repository
git clone <your-repo-url>.git
cd Text--SQL

# (optional) create a virtual environment
python -m venv .venv
# Windows
.venv\\Scripts\\activate

# install dependencies
pip install -r requirements.txt
```

`requirements.txt` currently includes at least:

- `torch`
- `numpy`

You can add others as needed (e.g. Jupyter, tqdm, etc.).

---

## Training

Main training entry point: `src/training/trainer.py`.

It will:

1. Build vocabularies from the training JSONL.
2. Create train and dev `DataLoader`s.
3. Build the Seq2Seq model.
4. Train for a configurable number of epochs with teacher forcing.
5. Evaluate on the dev set (loss).
6. Save the final checkpoint to `checkpoints/text2sql.pt`.

Run from the project root:

```bash
python -m src.training.trainer
```

Hyperparameters such as:

- `num_epochs`
- `batch_size`
- `learning_rate`
- `teacher_forcing_ratio`
- model dimensions (embedding size, hidden size, num layers, dropout)

are defined inside `trainer.py` in `main()` and in `build_model()`.

---

## Evaluation

A simple dev evaluation script (`eval_wikisql`) computes:

- Dev loss (from `trainer.py`).
- Exact match rate (predicted SQL string equals gold SQL string).
- Token-level accuracy (fraction of gold SQL tokens that match at correct positions).

The evaluation loops through `wikisql_dev.jsonl`, uses the trained model
checkpoint (`checkpoints/text2sql.pt`), and prints:

- A few sample predictions (question / gold SQL / predicted SQL).
- Aggregated metrics over the dev set.

Exact match is typically low for this simple LSTM model on full WikiSQL,
but the goal is to understand behavior and limitations, not to match
state-of-the-art results.

---

## Inference (Interactive)

For quick experiments, you can run:

```bash
python -m src.training.inference
```

This will:

1. Rebuild vocabularies from the training data.
2. Recreate the model architecture.
3. Load the checkpoint from `checkpoints/text2sql.pt`.
4. Open an interactive loop:

```text
Enter a question (or 'quit' to exit):
> What school did player number 21 play for?
Predicted SQL:
select school/club team from table where no. = ...
```

Internally, it uses `translate_question(...)`, which:

- Tokenizes a question.
- Runs the model with greedy decoding.
- Converts token IDs back to a SQL string with basic cleanup.

---

## Colab Training

Most experiments were run in Google Colab using a GPU.
A typical workflow:

1. Clone this repo in Colab and mount data (or upload prepared JSONL files).
2. Run `scripts/prepare_wikisql.py` if you need to regenerate the processed data.
3. Execute `python -m src.training.trainer` in a Colab cell to train.
4. Download the resulting `checkpoints/text2sql.pt` and keep it in this repo.
5. Run the local/eval scripts using that checkpoint.

You can commit the Colab notebook (e.g. `notebooks/wikisql_training.ipynb`)
that documents your experiments. Large checkpoints should usually not be
committed directly to Git.

---

## Limitations & Lessons

This project intentionally uses a classic LSTM Seq2Seq with attention:

- No pretraining (e.g. BERT).
- No schema-aware encoding of column names/types.
- No constrained decoding or SQL grammar.

As a result, the model often produces plausible SQL skeletons but incorrect details
(wrong columns or values, missing conditions, etc.). Dev exact-match accuracy
is limited, even after tuning.

This is expected and part of the learning experience:

- You see how far a simple Seq2Seq can go.
- You see why modern Text-to-SQL systems rely on pretrained transformers,
  schema encoding, and sometimes grammar-based decoding.

---

## Possible Next Steps (v2)

Potential extensions:

- Use a pretrained transformer encoder (e.g. BERT) for the question.
- Add schema-aware inputs (explicit table/column representations).
- Improve SQL tokenization and normalization.
- Add constrained or grammar-based decoding for valid SQL.

---

## Acknowledgements

- Dataset: WikiSQL and related Kaggle variants.
- Many design ideas are inspired by classic Seq2Seq tutorials and
  Text-to-SQL research; this implementation is written from scratch
  for learning purposes.
