from collections import Counter
from typing import List


class Vocabulary:
    def __init__(self, specials: List[str]) -> None:
        self.token_to_idx: dict[str, int] = {}
        self.idx_to_token: list[str] = []
        self.freqs: Counter[str] = Counter()

        # Add special tokens first
        for token in specials:
            self._add_special(token)

        # Cache the unk token index, assuming it's in specials
        self.unk_token = "<unk>"
        self.unk_idx = self.token_to_idx.get(self.unk_token, 0)

    def _add_special(self, token: str) -> None:
        if token not in self.token_to_idx:
            idx = len(self.idx_to_token)
            self.token_to_idx[token] = idx
            self.idx_to_token.append(token)

    def add_token(self, token: str) -> None:
        """
        Update frequency count for a token.
        We do NOT assign indices for normal tokens here; that happens in build_from_sequences.
        """
        self.freqs[token] += 1

    def build_from_sequences(self, sequences: List[List[str]], min_freq: int = 1) -> None:
        """
        Build vocabulary from token sequences.

        - Count frequencies of all tokens.
        - For tokens with freq >= min_freq and not already special, assign new indices.
        """
        # 1) Accumulate frequencies
        for seq in sequences:
            for token in seq:
                self.add_token(token)

        # 2) For each token (sorted for reproducibility), assign index if freq >= min_freq
        for token, freq in sorted(self.freqs.items()):
            if freq < min_freq:
                continue
            if token in self.token_to_idx:
                # Already a special token
                continue
            idx = len(self.idx_to_token)
            self.token_to_idx[token] = idx
            self.idx_to_token.append(token)

    def token_to_id(self, token: str) -> int:
        """
        Convert token string to integer ID, using <unk> for unknown tokens.
        """
        return self.token_to_idx.get(token, self.unk_idx)

    def id_to_token(self, idx: int) -> str:
        """
        Convert integer ID back to token string.
        """
        if 0 <= idx < len(self.idx_to_token):
            return self.idx_to_token[idx]
        # Fallback to <unk> if index is invalid
        return self.unk_token

    @property
    def size(self) -> int:
        return len(self.idx_to_token)