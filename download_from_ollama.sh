#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <VRAM size: 12, 24>"
    exit 1
fi

VRAM=$1

if [ "$VRAM" == 12 ]; then
    models=("llama3.2-vision:11b" "qwen2.5vl:7b" "gemma3:12b" "llava:13b")
    for model in "${models[@]}"; do
        ollama pull "$model"
    done
elif [ "$VRAM" == 24 ]; then
    models=("llama3.2-vision:11b" "qwen2.5vl:32b" "gemma3:27b" "llava:34b" "mistral-small3.1")
    for model in "${models[@]}"; do
        ollama pull "$model"
    done
    echo "Invalid VRAM size. Please enter 12, 24."
fi
