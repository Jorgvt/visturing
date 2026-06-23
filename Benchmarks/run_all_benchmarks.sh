#!/bin/bash

# Array of models to evaluate
models=("resnet18" "vgg16" "convnext" "vit" "clip" "dinov2")

echo "============================================="
echo "Starting Evaluation of All Vision Models"
echo "============================================="

for model in "${models[@]}"
do
    echo ""
    echo "---------------------------------------------"
    echo "Running evaluation for model: $model"
    echo "---------------------------------------------"
    
    # Run the evaluation using uv
    uv run Benchmarks/evaluate_vision_models.py --model "$model" --dataset all --batch-size 32
done

echo ""
echo "============================================="
echo "All Evaluations Completed!"
echo "Summary tables are available under Benchmarks/results/"
echo "============================================="
