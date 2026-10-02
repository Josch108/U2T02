"""
Sanity check script to verify evaluation pipeline against reference numbers (U2T02).
Reference values (STS-B, Spearman x 100):
  - raw bert-base-uncased with mean pooling: 59.31 dev / 47.29 test
  - SBERT-2019 (bert-base-nli-mean-tokens): 80.77 dev / 76.98 test
"""

import argparse
import torch
from transformers import AutoModel, AutoTokenizer
from src.data_loader import load_stsb_data
from src.evaluate import evaluate_stsb, evaluate_sentence_transformer


def main():
    parser = argparse.ArgumentParser(description="Verify STS-B evaluation on reference baselines")
    parser.add_argument("--model", type=str, default="raw_bert", choices=["raw_bert", "sbert_2019", "both"])
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    dev_stsb, test_stsb = load_stsb_data()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running evaluation sanity check on device: {device}")

    if args.model in ("raw_bert", "both"):
        print("\n==========================================")
        print("Evaluating: raw bert-base-uncased (mean pooling)")
        print("Reference: 59.31 Dev | 47.29 Test")
        print("==========================================")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").to(device)

        dev_res = evaluate_stsb(model, tokenizer, dev_stsb, batch_size=args.batch_size, device=device, pooling="mean")
        test_res = evaluate_stsb(model, tokenizer, test_stsb, batch_size=args.batch_size, device=device, pooling="mean")

        print(f"Observed Dev Spearman:  {dev_res['spearman']:.2f} (Target: 59.31)")
        print(f"Observed Test Spearman: {test_res['spearman']:.2f} (Target: 47.29)")
        print(f"Dev Alignment:  {dev_res.get('alignment', 0.0):.4f}")
        print(f"Dev Uniformity: {dev_res.get('uniformity', 0.0):.4f}")

    if args.model in ("sbert_2019", "both"):
        print("\n==========================================")
        print("Evaluating: SBERT-2019 (bert-base-nli-mean-tokens)")
        print("Reference: 80.77 Dev | 76.98 Test")
        print("==========================================")
        sbert_dev = evaluate_sentence_transformer("bert-base-nli-mean-tokens", dev_stsb)
        sbert_test = evaluate_sentence_transformer("bert-base-nli-mean-tokens", test_stsb)

        print(f"Observed Dev Spearman:  {sbert_dev['spearman']:.2f} (Target: 80.77)")
        print(f"Observed Test Spearman: {sbert_test['spearman']:.2f} (Target: 76.98)")
        print(f"Test Alignment:  {sbert_test.get('alignment', 0.0):.4f}")
        print(f"Test Uniformity: {sbert_test.get('uniformity', 0.0):.4f}")


if __name__ == "__main__":
    main()
