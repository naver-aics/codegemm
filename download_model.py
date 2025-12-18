from huggingface_hub import snapshot_download
import argparse
import os

def download_model(repo_id, local_dir=None, cache_dir=None):
    """
    Download a Hugging Face model to a local directory.
    
    Args:
        repo_id: Hugging Face model repository ID (e.g., "gunho1123/Llama-3.1-8B-Instruct-Codegemm-m2v8g128")
        local_dir: Local directory path to save the model (uses cache directory if None)
        cache_dir: Hugging Face cache directory
    
    Returns:
        str: Path to the downloaded model
    """
    print(f"Downloading model: {repo_id}")
    
    local_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
        local_dir_use_symlinks=False,  # Copy actual files instead of using symlinks
    )
    
    print(f"Model downloaded to: {local_path}")
    return local_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True,
                       help="Hugging Face model repository ID")
    parser.add_argument("--local_dir", type=str, default=None,
                       help="Local directory to save the model")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Hugging Face cache directory")
    
    args = parser.parse_args()
    download_model(args.repo_id, args.local_dir, args.cache_dir)