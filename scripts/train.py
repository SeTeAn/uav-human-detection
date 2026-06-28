from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model for UAV human detection.")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to YOLO dataset config.")
    parser.add_argument("--weights", default="yolo11m.pt", help="Initial YOLO weights or model name.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'.")
    parser.add_argument("--name", default="uav-human-detection-yolo11m")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.set_device(int(args.device))

    model = YOLO(args.weights)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        cos_lr=True,
        workers=4,
        save_period=10,
        patience=0,
        optimizer="AdamW",
        name=args.name,
        single_cls=True,
        label_smoothing=0.1,
        device=args.device,
    )


if __name__ == "__main__":
    main()
