"""
T5 Inference for Text-to-SQL

Load a fine-tuned T5 model and generate SQL from natural language questions.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List


class T5TextToSQLInference:
    """
    Inference wrapper for T5 Text-to-SQL model.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device = None,
        max_input_length: int = 128,
        max_output_length: int = 64,
    ):
        """
        Args:
            model_path: Path to saved T5 model directory.
            device: torch device (defaults to CUDA if available).
            max_input_length: Max tokens for input.
            max_output_length: Max tokens for generated output.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        print(f"Loading model from {model_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    @torch.no_grad()
    def predict(
        self,
        question: str,
        num_beams: int = 4,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate SQL from a natural language question.
        
        Args:
            question: Natural language question.
            num_beams: Beam search width (higher = better quality, slower).
            num_return_sequences: Number of SQL candidates to return.
        
        Returns:
            List of generated SQL strings.
        """
        # Format input like training data
        input_text = f"translate English to SQL: {question}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Generate
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_output_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
        )

        # Decode
        predictions = []
        for output in outputs:
            sql = self.tokenizer.decode(output, skip_special_tokens=True)
            predictions.append(sql)

        return predictions

    def predict_batch(
        self,
        questions: List[str],
        num_beams: int = 4,
    ) -> List[str]:
        """
        Generate SQL for a batch of questions.
        
        Args:
            questions: List of natural language questions.
            num_beams: Beam search width.
        
        Returns:
            List of generated SQL strings (one per question).
        """
        # Format inputs
        input_texts = [f"translate English to SQL: {q}" for q in questions]

        # Tokenize batch
        inputs = self.tokenizer(
            input_texts,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Generate
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_output_length,
            num_beams=num_beams,
            early_stopping=True,
        )

        # Decode
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions


def main():
    """Interactive inference CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="T5 Text-to-SQL Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/v2/t5_best.pt",
        help="Path to fine-tuned T5 model",
    )
    args = parser.parse_args()

    # Load model
    inference = T5TextToSQLInference(model_path=args.model_path)

    print("\nEnter a question (or 'quit' to exit):")
    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        predictions = inference.predict(question, num_beams=4, num_return_sequences=1)
        print(f"Predicted SQL: {predictions[0]}")
        print()


if __name__ == "__main__":
    main()
