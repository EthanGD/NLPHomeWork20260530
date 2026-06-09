@echo off
echo 🚀 Starting RAG Flask Server...
echo Usage: run.bat [model_arg]
echo   model_arg: 1 (original) | 2 (fine-tuned) | custom_path
echo.

set MODEL_ARG=%1
if "%MODEL_ARG%"=="" set MODEL_ARG=1

echo 🎯 Using model argument: %MODEL_ARG%
python app.py %MODEL_ARG%