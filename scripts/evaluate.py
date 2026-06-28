from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO checkpoint.")
    parser.add_argument("--weights", required=True, help="Path to a trained checkpoint, for example runs/detect/.../weights/best.pt.")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to YOLO dataset config.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'.")
    parser.add_argument("--output", default="results/metrics.txt", help="Where to save metric values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device)
    results = metrics.results_dict

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "Test metrics:",
                f"mAP@0.5: {results['metrics/mAP50(B)']}",
                f"mAP@0.5:0.95: {results['metrics/mAP50-95(B)']}",
                f"Precision: {results['metrics/precision(B)']}",
                f"Recall: {results['metrics/recall(B)']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
