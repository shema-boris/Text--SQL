"""
T5 Dataset for Text-to-SQL

Formats WikiSQL JSONL into T5 input/output pairs:
- Input:  "translate English to SQL: <question>"
- Output: "<sql_query>"

Optionally includes column names in input for schema awareness.
"""

import json
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from transformers import T5Tokenizer


class T5TextToSQLDataset(Dataset):
    """
    Dataset that prepares text-to-SQL examples for T5 fine-tuning.
    
    Each example becomes:
        input_text:  "translate English to SQL: What school did player 21 play for?"
        target_text: "SELECT School/Club Team FROM table WHERE No. = 21"
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: T5Tokenizer,
        max_input_length: int = 128,
        max_target_length: int = 128,
        include_columns: bool = False,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with {"question": ..., "sql": ...} lines.
            tokenizer: HuggingFace T5 tokenizer.
            max_input_length: Max tokens for input sequence.
            max_target_length: Max tokens for target sequence.
            include_columns: If True, append column names to input (requires "columns" field in JSONL).
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.include_columns = include_columns

        self.examples: List[Dict[str, str]] = []
        self._load_data(jsonl_path)

    def _load_data(self, path: str) -> None:
        """Load JSONL and format as T5 input/output pairs."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                question = obj["question"]
                sql = obj["sql"]

                # Format input for T5
                if self.include_columns and "columns" in obj:
                    cols_str = " | ".join(obj["columns"])
                    input_text = f"translate English to SQL: {question} | columns: {cols_str}"
                else:
                    input_text = f"translate English to SQL: {question}"

                target_text = sql

                self.examples.append({
                    "input_text": input_text,
                    "target_text": target_text,
                })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns tokenized input and target for T5.
        
        Returns dict with:
            - input_ids: tokenized input
            - attention_mask: attention mask for input
            - labels: tokenized target (with padding tokens set to -100 for loss masking)
        """
        example = self.examples[idx]

        # Tokenize input
        input_encoding = self.tokenizer(
            example["input_text"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            example["target_text"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token id with -100 so it's ignored in loss
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def load_t5_dataset(
    jsonl_path: str,
    tokenizer_name: str = "t5-base",
    max_input_length: int = 128,
    max_target_length: int = 128,
) -> T5TextToSQLDataset:
    """
    Convenience function to load a T5 dataset.
    
    Args:
        jsonl_path: Path to JSONL file.
        tokenizer_name: HuggingFace tokenizer name (e.g., "t5-base", "t5-small").
        max_input_length: Max input tokens.
        max_target_length: Max target tokens.
    
    Returns:
        T5TextToSQLDataset instance.
    """
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    return T5TextToSQLDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )
