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
    parser.add_argument("--img-size", type=int, default=1008,
                        help="Image size for inference (default: 1008, matches training)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Subset of seed models to ensemble (default: all trained seeds). "
                             "Use a single seed to produce one submission per seed.")
    parser.add_argument("--submission-file", type=str, default=None,
                        help="Output CSV path (default: submission.csv, or submission_seed_<N>.csv "
                             "when a single seed is selected)")
    args = parser.parse_args()

    config = InferenceConfig()
    config.img_size = args.img_size
    config.batch_size = args.batch_size
    config.seeds = args.seeds
    if args.submission_file is not None:
        config.submission_file = args.submission_file
    elif args.seeds is not None and len(args.seeds) == 1:
        config.submission_file = f"submission_seed_{args.seeds[0]}.csv"
    print(f"Inference config: img_size={config.img_size}, batch_size={config.batch_size}, "
          f"seeds={config.seeds}, device={config.device}")

    pipeline = InferencePipeline(config)
    pipeline.run()

    print("\n" + "=" * 70)
    print("Inference Pipeline Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
