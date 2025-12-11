from typing import List, Dict, Any
import torch


def text_to_sql_collate_fn(
    batch: List[Dict[str, Any]],
    pad_idx: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for TextToSQLDataset.

    batch: list of items from TextToSQLDataset.__getitem__, each is:
        {
            "src_ids": LongTensor (src_len_i,),
            "trg_ids": LongTensor (trg_len_i,),
            "src_len": LongTensor scalar,
            "trg_len": LongTensor scalar,
        }

    Returns:
        {
            "src_batch": LongTensor (batch_size, max_src_len),
            "trg_batch": LongTensor (batch_size, max_trg_len),
            "src_lengths": LongTensor (batch_size,),
            "trg_lengths": LongTensor (batch_size,),
        }
    """
    batch_size = len(batch)

    # Extract lengths
    src_lengths = torch.tensor(
        [item["src_len"].item() for item in batch],
        dtype=torch.long,
    )
    trg_lengths = torch.tensor(
        [item["trg_len"].item() for item in batch],
        dtype=torch.long,
    )

    max_src_len = src_lengths.max().item()
    max_trg_len = trg_lengths.max().item()

    # Initialize padded batches
    src_batch = torch.full(
        (batch_size, max_src_len),
        fill_value=pad_idx,
        dtype=torch.long,
    )
    trg_batch = torch.full(
        (batch_size, max_trg_len),
        fill_value=pad_idx,
        dtype=torch.long,
    )

    # Fill with actual ids
    for i, item in enumerate(batch):
        src_ids = item["src_ids"]
        trg_ids = item["trg_ids"]

        src_len = src_ids.size(0)
        trg_len = trg_ids.size(0)

        src_batch[i, :src_len] = src_ids
        trg_batch[i, :trg_len] = trg_ids

    return {
        "src_batch": src_batch,
        "trg_batch": trg_batch,
        "src_lengths": src_lengths,
        "trg_lengths": trg_lengths,
    }