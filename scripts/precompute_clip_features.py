#!/usr/bin/env python3
"""Pre-compute CLIP features for faster training - Optimized version."""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

from datasets import load_dataset
from transformers import CLIPProcessor, CLIPVisionModel


def download_image(args: Tuple[int, str]) -> Tuple[int, Optional[Image.Image]]:
    """
    Download image from URL with validation.

    Args:
        args: Tuple of (index, url)

    Returns:
        Tuple of (index, image or None)
    """
    idx, url = args
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # Convert to RGB
        image = image.convert("RGB")

        # Validate dimensions
        width, height = image.size
        if width < 10 or height < 10:
            return idx, None

        return idx, image
    except Exception:
        return idx, None


def download_batch_parallel(
    indices: List[int],
    urls: List[str],
    num_workers: int = 32,
) -> List[Tuple[int, Image.Image]]:
    """Download images in parallel."""
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_image, (idx, url)): idx
            for idx, url in zip(indices, urls)
        }

        for future in as_completed(futures):
            idx, image = future.result()
            if image is not None:
                results.append((idx, image))

    return results


def precompute_cc3m_features(
    output_dir: str = "./cache",
    subset_size: int = 100000,
    batch_size: int = 1024,  # Very large batch for A100 40GB
    download_workers: int = 128,  # More parallel downloads
    device: str = "cuda",
) -> None:
    """
    Pre-compute CLIP features for Conceptual Captions dataset.
    Optimized for A100 GPU with parallel downloading.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CLIP model
    print("Loading CLIP ViT-B/16...")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = model.to(device)
    model.eval()

    # Enable TF32 for A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load dataset
    print(f"Loading Conceptual Captions (subset: {subset_size})...")
    dataset = load_dataset("conceptual_captions", split="train")
    if subset_size < len(dataset):
        dataset = dataset.select(range(subset_size))

    print(f"Processing {len(dataset)} samples...")
    print(f"Batch size: {batch_size}, Download workers: {download_workers}")

    # Pre-allocate memory-mapped array
    features_path = output_dir / "clip_features.npy"
    features = np.memmap(
        features_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(dataset), 197, 768),
    )

    # Track valid indices
    valid_indices = {}
    current_valid = 0

    # Process in larger batches with parallel downloading
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(dataset), batch_size), desc="Computing features", total=num_batches)

    for start_idx in pbar:
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_indices = list(range(start_idx, end_idx))

        # Get URLs for this batch
        urls = [dataset[idx]["image_url"] for idx in batch_indices]

        # Download images in parallel
        downloaded = download_batch_parallel(batch_indices, urls, download_workers)

        if not downloaded:
            continue

        # Sort by index to maintain order
        downloaded.sort(key=lambda x: x[0])
        valid_batch_indices = [d[0] for d in downloaded]
        images = [d[1] for d in downloaded]

        try:
            # Process in sub-batches - 512 fits well on 40GB A100
            sub_batch_size = 512
            for sub_start in range(0, len(images), sub_batch_size):
                sub_end = min(sub_start + sub_batch_size, len(images))
                sub_images = images[sub_start:sub_end]
                sub_indices = valid_batch_indices[sub_start:sub_end]

                # Process images
                inputs = processor(images=sub_images, return_tensors="pt", padding=True)
                pixel_values = inputs["pixel_values"].to(device)

                # Extract features
                with torch.no_grad(), torch.cuda.amp.autocast():  # Use FP16
                    outputs = model(pixel_values)
                    batch_features = outputs.last_hidden_state.float().cpu().numpy()

                # Store features
                for i, idx in enumerate(sub_indices):
                    features[current_valid] = batch_features[i]
                    valid_indices[str(idx)] = current_valid
                    current_valid += 1

        except Exception as e:
            print(f"\nError processing batch {start_idx}-{end_idx}: {e}")
            continue

        # Update progress
        success_rate = current_valid / end_idx if end_idx > 0 else 0
        pbar.set_postfix({
            "valid": current_valid,
            "rate": f"{success_rate:.1%}",
            "gpu_mb": f"{torch.cuda.memory_allocated() / 1e6:.0f}"
        })

    # Flush
    features.flush()

    # Save index
    index_path = output_dir / "feature_index.json"
    with open(index_path, "w") as f:
        json.dump(valid_indices, f)

    print(f"\nDone! Processed {current_valid}/{len(dataset)} samples ({current_valid/len(dataset):.1%})")
    print(f"Features saved to: {features_path}")
    print(f"Index saved to: {index_path}")

    # Create properly sized array
    if current_valid < len(dataset):
        print(f"Resizing features array to {current_valid} samples...")
        final_features = np.memmap(
            output_dir / "clip_features_final.npy",
            dtype=np.float32,
            mode="w+",
            shape=(current_valid, 197, 768),
        )
        final_features[:] = features[:current_valid]
        final_features.flush()

        # Replace original
        import os
        os.remove(features_path)
        (output_dir / "clip_features_final.npy").rename(features_path)


def precompute_coco_features(
    output_dir: str = "./cache",
    batch_size: int = 128,
    download_workers: int = 64,
    device: str = "cuda",
) -> None:
    """Pre-compute CLIP features for COCO Captions validation set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP ViT-B/16...")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = model.to(device)
    model.eval()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading COCO Captions validation set...")
    # Use yerevann/coco-karpathy which has clean splits
    dataset = load_dataset("yerevann/coco-karpathy", split="test")

    # Get unique images with their URLs
    unique_images = {}
    for sample in tqdm(dataset, desc="Collecting unique images"):
        image_id = sample["cocoid"]
        if image_id not in unique_images:
            unique_images[image_id] = sample["url"]

    print(f"Processing {len(unique_images)} unique images...")

    features_path = output_dir / "coco_clip_features.npy"
    features = np.memmap(
        features_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(unique_images), 197, 768),
    )

    feature_index = {}
    image_ids = list(unique_images.keys())
    current_valid = 0

    # Process with parallel downloading
    num_batches = (len(image_ids) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(image_ids), batch_size), desc="Computing COCO features", total=num_batches)

    for start_idx in pbar:
        end_idx = min(start_idx + batch_size, len(image_ids))
        batch_ids = image_ids[start_idx:end_idx]
        batch_urls = [unique_images[img_id] for img_id in batch_ids]

        # Download images in parallel
        downloaded = download_batch_parallel(
            list(range(len(batch_ids))),  # Use local indices
            batch_urls,
            download_workers,
        )

        if not downloaded:
            continue

        # Map back to image IDs
        valid_ids = []
        images = []
        for local_idx, img in downloaded:
            valid_ids.append(batch_ids[local_idx])
            images.append(img)

        try:
            inputs = processor(images=images, return_tensors="pt", padding=True)
            pixel_values = inputs["pixel_values"].to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model(pixel_values)
                batch_features = outputs.last_hidden_state.float().cpu().numpy()

            for i, img_id in enumerate(valid_ids):
                features[current_valid] = batch_features[i]
                feature_index[str(img_id)] = current_valid
                current_valid += 1

        except Exception as e:
            print(f"\nError: {e}")
            continue

        pbar.set_postfix({"valid": current_valid})

    features.flush()

    index_path = output_dir / "coco_feature_index.json"
    with open(index_path, "w") as f:
        json.dump(feature_index, f)

    print(f"\nDone! Processed {current_valid}/{len(unique_images)} images")
    print(f"Features saved to: {features_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CLIP features (Optimized)")
    parser.add_argument("--dataset", type=str, default="cc3m", choices=["cc3m", "coco", "both"])
    parser.add_argument("--output-dir", type=str, default="./cache")
    parser.add_argument("--subset-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1024, help="Download batch size")
    parser.add_argument("--download-workers", type=int, default=128, help="Parallel download threads")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.dataset in ["cc3m", "both"]:
        precompute_cc3m_features(
            output_dir=args.output_dir,
            subset_size=args.subset_size,
            batch_size=args.batch_size,
            download_workers=args.download_workers,
            device=args.device,
        )

    if args.dataset in ["coco", "both"]:
        precompute_coco_features(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
