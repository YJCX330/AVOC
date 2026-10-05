import json
import argparse
from collections import Counter


def cal_accuracy(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)

    total = len(results)
    correct = 0
    invalid = 0
    valid_options = {"A", "B", "C", "D"}

    invalid_samples = []
    wrong_samples = []

    for item in results:
        gt = item["gt_answer"].strip()
        pred = item.get("pred_option", "").strip()

        if pred not in valid_options:
            invalid += 1
            invalid_samples.append({
                "id": item["id"],
                "gt": gt,
                "pred_option": pred,
                "pred_raw": item.get("pred_raw", "")[:120],
            })
            continue

        if pred == gt:
            correct += 1
        else:
            wrong_samples.append({
                "id": item["id"],
                "gt": gt,
                "pred_option": pred,
            })

    valid = total - invalid
    acc = correct / total * 100 if total > 0 else 0
    acc_valid = correct / valid * 100 if valid > 0 else 0

    print(f"{'=' * 50}")
    print(f"Results: {results_path}")
    print(f"{'=' * 50}")
    print(f"Total samples:    {total}")
    print(f"Correct:          {correct}")
    print(f"Wrong:            {len(wrong_samples)}")
    print(f"Invalid pred:     {invalid}")
    print(f"{'─' * 50}")
    print(f"Accuracy (all):   {acc:.2f}% ({correct}/{total})")
    print(f"Accuracy (valid): {acc_valid:.2f}% ({correct}/{valid})")
    print(f"{'─' * 50}")

    gt_dist = Counter(item["gt_answer"].strip() for item in results)
    pred_dist = Counter(item.get("pred_option", "").strip() for item in results)
    print("\nAnswer distribution:")
    print(f"  {'Option':<8} {'GT':>6} {'Pred':>6}")
    for opt in sorted(valid_options):
        print(f"  {opt:<8} {gt_dist.get(opt, 0):>6} {pred_dist.get(opt, 0):>6}")
    if invalid > 0:
        other_keys = [k for k in pred_dist if k not in valid_options]
        for k in sorted(other_keys):
            print(f"  {repr(k):<8} {'-':>6} {pred_dist[k]:>6}")

    if invalid_samples:
        print(f"\nInvalid predictions ({invalid}):")
        for s in invalid_samples[:10]:
            print(f"  id={s['id']}, gt={s['gt']}, pred_option={repr(s['pred_option'])}")
            print(f"    pred_raw: {s['pred_raw']}...")
        if invalid > 10:
            print(f"  ... and {invalid - 10} more")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy for evaluation results")
    parser.add_argument(
        "results_path",
        nargs="?",
        default="./results/lvomnibench.json",
        help="Path to results JSON file",
    )
    args = parser.parse_args()
    cal_accuracy(args.results_path)
