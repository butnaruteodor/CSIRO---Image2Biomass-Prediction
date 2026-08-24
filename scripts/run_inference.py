#!/usr/bin/env python3
"""
Run full inference pipeline to produce submission.csv.

Usage:
    python scripts/run_inference.py
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import InferenceConfig
from src.inference.pipeline import InferencePipeline


def main():
    parser = argparse.ArgumentParser(description="Run biomass inference pipeline")
    parser.add_argument("--img-size", type=int, default=768,
                        help="Image size for inference (default: 768)")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = InferenceConfig()
    config.img_size = args.img_size
    config.batch_size = args.batch_size
    config.display_info()

    pipeline = InferencePipeline(config)
    pipeline.run()

    print("\n" + "=" * 70)
    print("Inference Pipeline Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
