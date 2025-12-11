
import json
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from .tokenizer import Vocabulary

@dataclass
class TextToSQLExample:
    src_tokens: List[str]
    trg_tokens: List[str]

class TextToSQLDataset(Dataset):
    def __init__(
        self,
        path:str,
        src_vocab: Vocabulary,
        trg_vocab:Vocabulary,
        max_src_len: int=64,
        max_trg_len: int=64,
    ) -> None:
        super().__init__()
        """
        path: path to a JSONL file with lines like:
            {"question": "...", "sql": "..."}
        """

        self.src_vocab=src_vocab
        self.trg_vocab = trg_vocab
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len
        self.examples: List[TextToSQLExample] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj=json.loads(line)
                question=obj["question"]
                sql=obj["sql"]
                src_tokens=self.tokenize_question(question)
                trg_tokens=self.tokenize_sql(sql)
                self.examples.append(TextToSQLExample(src_tokens, trg_tokens))
    def tokenize_question(self, text:str):
        return text.lower().strip().split()
    def tokenize_sql(self, text:str):
        return text.lower().strip().split()
    def __len__(self):
        return len(self.examples)
    def _truncate_and_add_specials(
        self,
        tokens:List[str],
        vocab: Vocabulary,
        add_sos:bool,
        add_eos: bool,
        max_len:int,
    )-> torch.Tensor:
        """
        Truncate, optionally add <sos>/<eos>, and convert to ids (no padding here).
        """
        if add_sos:
            tokens=["<sos>"]+tokens
        if add_eos:
            tokens=tokens+["<eos>"]
        tokens = tokens[:max_len]
        ids = [vocab.token_to_id(tok) for tok in tokens]
        return torch.tensor(ids, dtype=torch.long)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        # Source: no sos/eos, just truncate + ids
        src_ids = self._truncate_and_add_specials(
            tokens=example.src_tokens,
            vocab=self.src_vocab,
            add_sos=False,
            add_eos=False,
            max_len=self.max_src_len,
        )
        # Target: add <sos> and <eos>
        trg_ids = self._truncate_and_add_specials(
            tokens=example.trg_tokens,
            vocab=self.trg_vocab,
            add_sos=True,
            add_eos=True,
            max_len=self.max_trg_len,
        )
        return {
            "src_ids": src_ids,
            "trg_ids": trg_ids,
            "src_len": torch.tensor(len(src_ids), dtype=torch.long),
            "trg_len": torch.tensor(len(trg_ids), dtype=torch.long),
        }