"""
T5 Fine-tuning Trainer for Text-to-SQL

Fine-tunes a pretrained T5 model on WikiSQL question→SQL pairs.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Optional

from src.v2.dataset import T5TextToSQLDataset


def train_one_epoch(
    model: T5ForConditionalGeneration,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(data_loader)


@torch.no_grad()
def evaluate(
    model: T5ForConditionalGeneration,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate on dev set.
    
    Returns:
        Average loss over the dataset.
    """
    model.eval()
    total_loss = 0.0

    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        total_loss += outputs.loss.item()

    return total_loss / len(data_loader)


def main():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    train_path = "data/raw/wikisql_train.jsonl"
    dev_path = "data/raw/wikisql_dev.jsonl"
    checkpoint_dir = "checkpoints/v2"

    # Model
    model_name = "t5-base"  # Can use "t5-small" for faster training

    # Training hyperparameters
    batch_size = 8          # Reduce if you hit OOM
    num_epochs = 3          # T5 converges fast; 3-5 epochs is often enough
    learning_rate = 3e-5    # Lower LR for fine-tuning pretrained models
    warmup_steps = 500
    max_input_length = 128
    max_target_length = 64

    # -------------------------------------------------------------------------
    # Load tokenizer and model
    # -------------------------------------------------------------------------
    print(f"Loading {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    train_dataset = T5TextToSQLDataset(
        jsonl_path=train_path,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    dev_dataset = T5TextToSQLDataset(
        jsonl_path=dev_path,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Dev examples: {len(dev_dataset)}")

    # -------------------------------------------------------------------------
    # Optimizer and scheduler
    # -------------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print("Starting training...")
    best_dev_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        dev_loss = evaluate(
            model=model,
            data_loader=dev_loader,
            device=device,
        )

        print(f"Epoch {epoch}: train_loss = {train_loss:.4f}, dev_loss = {dev_loss:.4f}")

        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_path = f"{checkpoint_dir}/t5_best.pt"
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved best model to {save_path}")

    # -------------------------------------------------------------------------
    # Save final model
    # -------------------------------------------------------------------------
    final_path = f"{checkpoint_dir}/t5_final.pt"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved final model to {final_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
