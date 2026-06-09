#!/bin/bash
echo "🚀 Starting RAG Flask Server..."
echo "Usage: ./run.sh [model_arg]"
echo "  model_arg: 1 (original) | 2 (fine-tuned) | /custom/path"
echo

MODEL_ARG=${1:-1}
echo "🎯 Using model argument: $MODEL_ARG"

python app.py "$MODEL_ARG"