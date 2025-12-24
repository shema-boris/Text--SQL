"""
Evaluation script for T5 Text-to-SQL model.

Computes:
- Exact match accuracy
- Token-level accuracy
- Prints sample predictions for qualitative analysis
"""

import json
from typing import List, Tuple
from tqdm import tqdm

from src.v2.inference import T5TextToSQLInference


def exact_match(gold: str, pred: str) -> float:
    """Returns 1.0 if gold == pred (after strip), else 0.0."""
    return 1.0 if gold.strip() == pred.strip() else 0.0


def token_accuracy(gold: str, pred: str) -> float:
    """
    Simple token-level accuracy.
    Compares tokens position-by-position.
    """
    gold_tokens = gold.strip().split()
    pred_tokens = pred.strip().split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0

    matches = sum(
        1 for i, g in enumerate(gold_tokens)
        if i < len(pred_tokens) and pred_tokens[i] == g
    )
    return matches / len(gold_tokens)


def evaluate_model(
    inference: T5TextToSQLInference,
    dev_path: str,
    num_examples_to_print: int = 10,
    max_examples: int = None,
) -> Tuple[float, float]:
    """
    Evaluate the model on a dev set.
    
    Args:
        inference: T5TextToSQLInference instance.
        dev_path: Path to dev JSONL file.
        num_examples_to_print: How many examples to print for inspection.
        max_examples: Max examples to evaluate (None = all).
    
    Returns:
        Tuple of (exact_match_rate, token_accuracy_rate).
    """
    examples = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            examples.append(json.loads(line))

    if max_examples:
        examples = examples[:max_examples]

    total_em = 0.0
    total_tok_acc = 0.0
    n = 0
    printed = 0

    print(f"\nEvaluating on {len(examples)} examples...")
    for ex in tqdm(examples, desc="Evaluating"):
        question = ex["question"]
        gold_sql = ex["sql"]

        # Get prediction
        pred_sql = inference.predict(question, num_beams=4)[0]

        # Compute metrics
        em = exact_match(gold_sql, pred_sql)
        tok_acc = token_accuracy(gold_sql, pred_sql)

        total_em += em
        total_tok_acc += tok_acc
        n += 1

        # Print some examples
        if printed < num_examples_to_print:
            print("=" * 80)
            print(f"Example {printed + 1}")
            print(f"QUESTION : {question}")
            print(f"GOLD SQL : {gold_sql}")
            print(f"PRED SQL : {pred_sql}")
            print(f"EXACT MATCH: {em:.0f}")
            print(f"TOKEN ACC  : {tok_acc:.3f}")
            printed += 1

    em_rate = total_em / n if n > 0 else 0.0
    tok_acc_rate = total_tok_acc / n if n > 0 else 0.0

    print("-" * 80)
    print(f"Evaluated on {n} examples")
    print(f"Exact Match Rate     : {em_rate:.4f} ({em_rate*100:.2f}%)")
    print(f"Token-level Accuracy : {tok_acc_rate:.4f} ({tok_acc_rate*100:.2f}%)")

    return em_rate, tok_acc_rate


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate T5 Text-to-SQL model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/v2/t5_best.pt",
        help="Path to fine-tuned T5 model",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="data/raw/wikisql_dev.jsonl",
        help="Path to dev JSONL file",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Max examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--num_print",
        type=int,
        default=10,
        help="Number of examples to print",
    )
    args = parser.parse_args()

    # Load model
    inference = T5TextToSQLInference(model_path=args.model_path)

    # Evaluate
    evaluate_model(
        inference=inference,
        dev_path=args.dev_path,
        num_examples_to_print=args.num_print,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
