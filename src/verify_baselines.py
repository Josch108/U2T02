"""
Sanity-check the STS-B evaluator against assignment reference values.

By default only STS-B dev is evaluated. Use --include_test for the final
benchmark phase; when test is included, the script also saves report-ready
baseline geometry metrics.
"""

import argparse
import json
import os

import torch
from transformers import AutoModel, AutoTokenizer

from src.data_loader import load_stsb_data
from src.evaluate import evaluate_sentence_transformer, evaluate_stsb


RAW_BERT_DEV = 59.31
RAW_BERT_TEST = 47.29
SBERT_DEV = 80.77
SBERT_TEST = 76.98


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["raw_bert", "sbert_2019", "both"],
        default="both",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--include_test", action="store_true")
    parser.add_argument(
        "--output_json",
        default="report/artifacts/baselines.json",
        help="Written only when --include_test is used.",
    )
    args = parser.parse_args()

    dev_data, test_data = load_stsb_data()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print("Development-set sanity check")

    results = {}

    if args.model in ("raw_bert", "both"):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").to(device)

        dev = evaluate_stsb(
            model,
            tokenizer,
            dev_data,
            batch_size=args.batch_size,
            device=device,
            pooling="mean",
        )
        print(
            f"Raw BERT dev: {dev['spearman']:.2f} "
            f"(assignment reference {RAW_BERT_DEV:.2f})"
        )

        entry = {
            "dev_spearman": dev["spearman"],
            "assignment_reference_dev": RAW_BERT_DEV,
        }

        if args.include_test:
            test = evaluate_stsb(
                model,
                tokenizer,
                test_data,
                batch_size=args.batch_size,
                device=device,
                pooling="mean",
            )
            print(
                f"Raw BERT test: {test['spearman']:.2f} "
                f"(assignment reference {RAW_BERT_TEST:.2f})"
            )
            entry.update(
                {
                    "test_spearman": test["spearman"],
                    "test_alignment": test["alignment"],
                    "test_uniformity": test["uniformity"],
                    "assignment_reference_test": RAW_BERT_TEST,
                }
            )
        results["raw_bert"] = entry

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.model in ("sbert_2019", "both"):
        dev = evaluate_sentence_transformer(
            "sentence-transformers/bert-base-nli-mean-tokens",
            dev_data,
        )
        print(
            f"SBERT-2019 dev: {dev['spearman']:.2f} "
            f"(assignment reference {SBERT_DEV:.2f})"
        )

        entry = {
            "dev_spearman": dev["spearman"],
            "assignment_reference_dev": SBERT_DEV,
        }

        if args.include_test:
            test = evaluate_sentence_transformer(
                "sentence-transformers/bert-base-nli-mean-tokens",
                test_data,
            )
            print(
                f"SBERT-2019 test: {test['spearman']:.2f} "
                f"(assignment reference {SBERT_TEST:.2f})"
            )
            entry.update(
                {
                    "test_spearman": test["spearman"],
                    "test_alignment": test["alignment"],
                    "test_uniformity": test["uniformity"],
                    "assignment_reference_test": SBERT_TEST,
                }
            )
        results["sbert_2019"] = entry

    if args.include_test:
        write_json(args.output_json, results)
        print(f"Saved final baseline metrics to {args.output_json}")


if __name__ == "__main__":
    main()
