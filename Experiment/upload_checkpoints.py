#!/usr/bin/env python3
"""
Upload model checkpoints to Hugging Face Hub.

Usage:
    python upload_checkpoints.py [--repo-id REPO_ID] [--token TOKEN] [--dry-run]
"""

import argparse
from pathlib import Path
from typing import Dict

try:
    from huggingface_hub import HfApi, login, create_repo, upload_folder
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")


def find_modality_folders(checkpoints_dir: Path) -> Dict[str, Path]:
    """Find all modality folders containing .pt files."""
    modality_folders = {}
    
    for file_path in Path(checkpoints_dir).rglob('*.pt'):
        if file_path.is_file():
            current = file_path.parent
            while current != checkpoints_dir and current != checkpoints_dir.parent:
                if current.name.startswith('modalities_'):
                    if current.name not in modality_folders:
                        modality_folders[current.name] = current
                    break
                current = current.parent
    
    return modality_folders


def upload_checkpoints(checkpoints_dir: Path, repo_id: str, token: str = None, dry_run: bool = False):
    """Upload all checkpoints to Hugging Face Hub, organized by modality."""
    if not checkpoints_dir.exists():
        raise ValueError(f"Checkpoints directory does not exist: {checkpoints_dir}")
    
    modality_folders = find_modality_folders(checkpoints_dir)
    if not modality_folders:
        print("No modality folders with checkpoint files found.")
        return
    
    print(f"Found {len(modality_folders)} modality folders")
    
    if dry_run:
        for modality, folder_path in modality_folders.items():
            pt_files = list(folder_path.rglob('*.pt'))
            total_size = sum(f.stat().st_size for f in pt_files) / (1024 * 1024)
            print(f"  {modality}: {len(pt_files)} .pt files ({total_size:.2f} MB)")
        return
    
    # Authenticate
    api = HfApi()
    if token:
        login(token=token)
        api.token = token
    else:
        # Use cached token
        try:
            api.whoami()
        except Exception:
            raise RuntimeError("Not authenticated. Run 'huggingface-cli login' or provide --token")
    
    # Ensure repository exists
    try:
        api.repo_info(repo_id, repo_type="model", token=api.token)
    except HfHubHTTPError as e:
        if e.status_code == 404:
            create_repo(repo_id=repo_id, repo_type="model", exist_ok=False, token=api.token)
        elif e.status_code == 403:
            raise RuntimeError(
                "Permission denied. Your token needs WRITE permissions.\n"
                "Create a write token at: https://huggingface.co/settings/tokens"
            )
        else:
            raise
    
    print(f"Uploading to {repo_id}...")
    
    for modality, folder_path in modality_folders.items():
        print(f"Uploading {modality}...")
        try:
            rel_path = folder_path.relative_to(checkpoints_dir)
            path_in_repo = str(rel_path).replace('\\', '/')
            
            upload_folder(
                folder_path=str(folder_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                token=api.token,
                ignore_patterns=["*.json", "*.png"],  # Only upload .pt files
            )
        except HfHubHTTPError as e:
            if e.status_code == 403:
                raise RuntimeError(
                    f"Permission denied for {modality}.\n"
                    "Your token needs WRITE permissions. "
                    "Create a write token at: https://huggingface.co/settings/tokens"
                )
            raise
    
    print(f"\n✓ Upload complete: {len(modality_folders)} modalities uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload model checkpoints to Hugging Face Hub")
    parser.add_argument("--repo-id", default="bibbbu/SRI-PD-v1", help="Hugging Face repository ID")
    parser.add_argument("--token", default=None, help="Hugging Face token (uses cached if not provided)")
    parser.add_argument("--checkpoints-dir", default=None, help="Path to checkpoints directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be uploaded")
    
    args = parser.parse_args()
    
    # Determine checkpoints directory
    if args.checkpoints_dir:
        checkpoints_dir = Path(args.checkpoints_dir)
    else:
        script_dir = Path(__file__).parent
        checkpoints_dir = script_dir / "V1_implementation" / "models" / "checkpoints"
    
    upload_checkpoints(
        checkpoints_dir=checkpoints_dir.resolve(),
        repo_id=args.repo_id,
        token=args.token,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
