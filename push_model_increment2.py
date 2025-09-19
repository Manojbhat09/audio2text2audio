import re
import json
import argparse
import os
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, move_repo
import subprocess

# Configuration
MODELS_JSON_FILE = "models.json"
incrementby = 1

def load_models_from_json(json_file: str) -> list[dict]:
    """Load models from JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"JSON file {json_file} not found. Creating with sample data...")
        return create_sample_models_json(json_file)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {json_file}: {e}")
        raise

def create_sample_models_json(json_file: str) -> list[dict]:
    """Create sample models JSON file with the original data."""
    sample_models = [
    {"wallet_name": "test2m3b2", "wallet_hotkey": "t2m3b25", "hotkey_address": "5EqHGDbh1DjE3LHWfBEdSNsP6DmWHzuwcP86wjHT81ubf5cy", "repo_namespace": "OwOpeepeepoopoo", "repo_id": "OwOpeepeepoopoo/imhungry-returnEqH-01"},

    {"wallet_name": "test2m3b2", "wallet_hotkey": "t2m3b27", "hotkey_address": "5G4aypoqAPqhTvUVYBEd8P9c8q6UzELQFdBamke8yS1GCs1T", "repo_namespace": "OwOpeepeepoopoo", "repo_id": "OwOpeepeepoopoo/imhungry-return2-01"},

    {"wallet_name": "test2m3b2", "wallet_hotkey": "t2m3b213", "hotkey_address": "5CFsFJ1fVX4V7xSzrY9SVMuU8CWY9YqQ9Mgmqh3dZPGrcfet", "repo_namespace": "OwOpeepeepoopoo", "repo_id": "OwOpeepeepoopoo/imhungry-return1-01"}
    ]
    
    save_models_to_json(sample_models, json_file)
    return sample_models

def save_models_to_json(models: list[dict], json_file: str) -> None:
    """Save models to JSON file."""
    with open(json_file, 'w') as f:
        json.dump(models, f, indent=2)

def save_failed_models(failed_models: list[dict], json_file: str) -> None:
    """Save failed models to a separate JSON file for retry."""
    failed_file = json_file.replace('.json', '_failed.json')
    with open(failed_file, 'w') as f:
        json.dump(failed_models, f, indent=2)
    print(f"Failed models saved to {failed_file} for retry")

def retry_failed_models(json_file: str) -> list[dict]:
    """Load failed models from retry file."""
    failed_file = json_file.replace('.json', '_failed.json')
    try:
        with open(failed_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No failed models file found at {failed_file}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing failed models file {failed_file}: {e}")
        return []

def increment_repo_suffix(repo):
    # Matches a trailing number and increments it
    match = re.search(r"(.*?)(\d+)$", repo)
    if match:
        prefix, num = match.groups()
        new_num = str(int(num) + incrementby).zfill(len(num))
        return prefix + new_num
    else:
        # No trailing number found, just append increment
        return repo + str(incrementby)

def process_models(models: list[dict], dry_run: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    """Process models for increment and upload. Returns (successful, failed_moves, failed_uploads)."""
    # Display current and new repo names
    print(f"{'Wallet':<12} {'Hotkey':<8} {'Old Repo':<30} {'New Repo':<30}")
    print("-"*80)

    for model in models:
        old_repo = model["repo_id"].split("/", 1)[1]
        new_repo = increment_repo_suffix(old_repo)
        model["new_repo_id"] = f"{model['repo_namespace']}/{new_repo}"
        print(f"{model['wallet_name']:<12} {model['wallet_hotkey']:<8} {model['repo_id']:<30} {model['new_repo_id']:<30}")

    if dry_run:
        print("\n[DRY RUN] Would process the above models. Use --no-dry-run to execute.")
        return [], [], []

    # Initialize HuggingFace API
    api = HfApi()

    print("\nMoving models to incremented repo names...")
    successful_moves = []
    failed_moves = []
    
    for model in models:
        print(f"Moving {model['repo_id']} -> {model['new_repo_id']} ...")
        try:
            move_repo(from_id=model["repo_id"], to_id=model["new_repo_id"], token=os.getenv("HF_ACCESS_TOKEN"))
            print("✓ Moved successfully.")
            successful_moves.append(model)
        except Exception as e:
            print(f"✗ Failed to move: {e}")
            failed_moves.append(model)

    print(f"\nMove results: {len(successful_moves)} successful, {len(failed_moves)} failed")

    print("\nRunning upload commands to push changes to chain...")
    successful_uploads = []
    failed_uploads = []
    
    for model in successful_moves:
        cmd = [
            "python", "miner_utils/upload_model.py",
            f"--hf_repo_id={model['new_repo_id']}",
            f"--wallet.name={model['wallet_name']}",
            f"--wallet.hotkey={model['wallet_hotkey']}",
            "--netuid", "21",
            "--competition_id", "v3"
        ]
        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("✓ Upload command finished successfully.")
            successful_uploads.append(model)
        except subprocess.CalledProcessError as e:
            print(f"✗ Upload command failed: {e}")
            failed_uploads.append(model)

    print(f"\nUpload results: {len(successful_uploads)} successful, {len(failed_uploads)} failed")
    
    return successful_uploads, failed_moves, failed_uploads

def main():
    """Main function to process models increment and upload."""
    parser = argparse.ArgumentParser(description="Increment and upload models with failure handling")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only previously failed models")
    parser.add_argument("--models-file", default=MODELS_JSON_FILE, help="Path to models JSON file")
    
    args = parser.parse_args()
    
    # Load models
    if args.retry_failed:
        models = retry_failed_models(args.models_file)
        if not models:
            print("No failed models to retry.")
            return
        print(f"Retrying {len(models)} previously failed models...")
    else:
        models = load_models_from_json(args.models_file)
    
    # Process models
    successful_uploads, failed_moves, failed_uploads = process_models(models, args.dry_run)
    
    if args.dry_run:
        return
    
    # Handle results
    all_failed = failed_moves + failed_uploads
    
    if successful_uploads:
        print(f"\n✓ Updating {args.models_file} with new repo IDs for {len(successful_uploads)} successful uploads...")
        # Load current models to update
        current_models = load_models_from_json(args.models_file)
        
        # Update successful models
        for success_model in successful_uploads:
            for current_model in current_models:
                if (current_model["wallet_name"] == success_model["wallet_name"] and 
                    current_model["wallet_hotkey"] == success_model["wallet_hotkey"]):
                    current_model["repo_id"] = success_model["new_repo_id"]
                    break
        
        save_models_to_json(current_models, args.models_file)
        print(f"✓ Successfully updated {args.models_file}")
    
    if all_failed:
        print(f"\n✗ {len(all_failed)} models failed. Saving for retry...")
        save_failed_models(all_failed, args.models_file)
        print("You can retry failed models with: python push_models_increment.py --retry-failed")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"✓ Successful uploads: {len(successful_uploads)}")
    print(f"✗ Failed moves: {len(failed_moves)}")
    print(f"✗ Failed uploads: {len(failed_uploads)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
