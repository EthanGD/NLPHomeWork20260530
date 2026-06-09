from huggingface_hub import snapshot_download
import os

model_id = "Qwen/Qwen3.5-0.8B-Base"
download_path = r"\NLP\v2\models"

os.makedirs(download_path, exist_ok=True)

snapshot_download(
    repo_id=model_id,
    local_dir=download_path,
    local_dir_use_symlinks=False
)
