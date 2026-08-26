#!/bin/bash
cd /home/teo/repos/CSIRO---Image2Biomass-Prediction || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
venv/bin/python scripts/extract_embeddings.py --mode test > /tmp/embed_test_final.log 2>&1
echo "EXIT_CODE=$?" >> /tmp/embed_test_final.log