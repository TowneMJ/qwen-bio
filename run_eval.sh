#!/bin/bash

# Quick eval runner for Qwen biology experiments

MODEL=${1:-"Qwen/Qwen3-4B-Instruct-2507"}
OUTPUT_NAME=${2:-"qwen3-4b-baseline"}

lm_eval --model vllm \
    --model_args pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.8 \
    --tasks mmlu_pro_biology \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path ./baseline_results/$OUTPUT_NAME \
    --log_samples

echo "Results saved to ./baseline_results/$OUTPUT_NAME"