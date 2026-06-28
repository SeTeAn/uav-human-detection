from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on images and save annotated outputs.")
    parser.add_argument("--weights", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--source", default="data/test/images", help="Image file or directory.")
    parser.add_argument("--output", default="results/predictions", help="Directory for annotated images.")
    parser.add_argument("--conf", type=float, default=0.3)
    return parser.parse_args()


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path
        for path in source.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    for image_path in iter_images(source):
        result = model(str(image_path), conf=args.conf)[0]
        annotated = result.plot()
        target = output_dir / image_path.name
        cv2.imwrite(str(target), annotated)
        print(f"Saved {target}")


if __name__ == "__main__":
    main()
