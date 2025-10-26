#!/bin/bash
# Quick NCCL test - should finish in < 1 minute if NCCL works

source ~/miniconda3/bin/activate diffsynth-s2v
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Testing NCCL with 4 GPUs (should finish in <1 min)..."

torchrun --nproc_per_node=4 test_nccl.py

echo "If you see 'NCCL test PASSED' above, NCCL works!"
echo "If it hangs or times out, NCCL is broken."
