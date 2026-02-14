"""
Generate ground truth label files from OGB MolBACE dataset.

This script creates:
- test_labels.csv: Ground truth labels for the test set
- valid_labels.csv: Ground truth labels for the validation set

These files should be stored in a PRIVATE repository and NOT committed to the public repo.

Usage:
    python scripts/generate_labels.py
    python scripts/generate_labels.py --output-dir /path/to/output
    python scripts/generate_labels.py --for-friend friend_name
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
import zipfile
import hashlib
from datetime import datetime

# Fix for PyTorch 2.6+ compatibility with torch_geometric
import torch.serialization
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except ImportError:
    pass

from ogb.graphproppred import PygGraphPropPredDataset


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def create_labels_package(
    output_dir: str,
    recipient_name: Optional[str] = None,
    include_train: bool = False,
    create_zip: bool = True
) -> dict:
    """
    Generate all label files and optionally package them.
    
    Args:
        output_dir: Directory to save labels
        recipient_name: Name of person receiving labels (for tracking)
        include_train: Whether to include training labels
        create_zip: Whether to create a zip archive
        
    Returns:
        Dictionary with file paths and metadata
    """
    print("Loading OGB MolBACE dataset...")
    
    # Determine data root
    script_dir = Path(__file__).parent
    data_root = script_dir.parent / "data" / "ogb"
    
    dataset = PygGraphPropPredDataset(name='ogbg-molbace', root=str(data_root))
    split_idx = dataset.get_idx_split()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    result = {
        'files': [],
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'recipient': recipient_name or 'unknown',
            'dataset': 'ogbg-molbace'
        }
    }
    
    # Generate TEST labels
    print("\n📋 Generating test labels...")
    test_idx = split_idx['test'].tolist()
    test_labels = [dataset[i].y.item() for i in test_idx]  # type: ignore[union-attr]
    test_df = pd.DataFrame({'id': test_idx, 'target': test_labels})
    
    test_path = os.path.join(output_dir, 'test_labels.csv')
    test_df.to_csv(test_path, index=False)
    
    print(f"  - Test set: {len(test_df)} molecules")
    print(f"  - Class distribution: {test_df['target'].value_counts().to_dict()}")
    
    result['files'].append({
        'name': 'test_labels.csv',
        'path': test_path,
        'count': len(test_df),
        'hash': compute_file_hash(test_path),
        'privacy': 'PRIVATE'
    })
    
    # Generate VALIDATION labels
    print("\n📋 Generating validation labels...")
    valid_idx = split_idx['valid'].tolist()
    valid_labels = [dataset[i].y.item() for i in valid_idx]  # type: ignore[union-attr]
    valid_df = pd.DataFrame({'id': valid_idx, 'target': valid_labels})
    
    valid_path = os.path.join(output_dir, 'valid_labels.csv')
    valid_df.to_csv(valid_path, index=False)
    
    print(f"  - Validation set: {len(valid_df)} molecules")
    print(f"  - Class distribution: {valid_df['target'].value_counts().to_dict()}")
    
    result['files'].append({
        'name': 'valid_labels.csv',
        'path': valid_path,
        'count': len(valid_df),
        'hash': compute_file_hash(valid_path),
        'privacy': 'PRIVATE'
    })
    
    # Generate TRAIN labels (optional)
    if include_train:
        print("\n📋 Generating train labels (for reference)...")
        train_idx = split_idx['train'].tolist()
        train_labels = [dataset[i].y.item() for i in train_idx]  # type: ignore[union-attr]
        train_df = pd.DataFrame({'id': train_idx, 'target': train_labels})
        
        train_path = os.path.join(output_dir, 'train_labels.csv')
        train_df.to_csv(train_path, index=False)
        
        print(f"  - Train set: {len(train_df)} molecules")
        print(f"  - Class distribution: {train_df['target'].value_counts().to_dict()}")
        
        result['files'].append({
            'name': 'train_labels.csv',
            'path': train_path,
            'count': len(train_df),
            'hash': compute_file_hash(train_path),
            'privacy': 'PUBLIC'
        })
    
    # Create metadata file
    metadata_path = os.path.join(output_dir, 'labels_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("ENIGMA Competition - Ground Truth Labels\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {result['metadata']['generated_at']}\n")
        f.write(f"Recipient: {result['metadata']['recipient']}\n")
        f.write(f"Dataset: {result['metadata']['dataset']}\n\n")
        f.write("Files:\n")
        for file_info in result['files']:
            f.write(f"  - {file_info['name']}: {file_info['count']} samples ")
            f.write(f"(hash: {file_info['hash']}, {file_info['privacy']})\n")
        f.write("\n")
        f.write("⚠️  IMPORTANT:\n")
        f.write("  - test_labels.csv and valid_labels.csv are PRIVATE\n")
        f.write("  - Do NOT share these files publicly\n")
        f.write("  - Do NOT commit these to public repositories\n")
    
    result['files'].append({
        'name': 'labels_metadata.txt',
        'path': metadata_path,
        'privacy': 'INFO'
    })
    
    # Create ZIP archive
    if create_zip:
        recipient_suffix = f"_{recipient_name}" if recipient_name else ""
        date_suffix = datetime.now().strftime("%Y%m%d")
        zip_name = f"enigma_labels{recipient_suffix}_{date_suffix}.zip"
        zip_path = os.path.join(output_dir, zip_name)
        
        print(f"\n📦 Creating ZIP archive: {zip_name}")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_info in result['files']:
                zf.write(file_info['path'], os.path.basename(file_info['path']))
        
        result['zip_path'] = zip_path
        print(f"  Archive created: {zip_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth labels for ENIGMA competition"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for labels (default: ../temp_private_data)'
    )
    parser.add_argument(
        '--for-friend', '-f',
        type=str,
        default=None,
        help='Name of friend/tester receiving these labels (for tracking)'
    )
    parser.add_argument(
        '--include-train',
        action='store_true',
        help='Include training labels (can be public)'
    )
    parser.add_argument(
        '--no-zip',
        action='store_true',
        help='Do not create ZIP archive'
    )
    parser.add_argument(
        '--copy-to-data',
        action='store_true',
        help='Also copy test_labels.csv to data/ for local testing'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    script_dir = Path(__file__).parent
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(script_dir.parent / 'temp_private_data')
    
    print("\n" + "=" * 60)
    print("  ENIGMA Label Generator")
    print("=" * 60)
    
    # Generate labels
    result = create_labels_package(
        output_dir=output_dir,
        recipient_name=args.for_friend,
        include_train=args.include_train,
        create_zip=not args.no_zip
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n✅ All label files saved to: {os.path.abspath(output_dir)}")
    print("\nFiles created:")
    for file_info in result['files']:
        privacy_icon = "🔒" if file_info['privacy'] == 'PRIVATE' else "📄"
        print(f"  {privacy_icon} {file_info['name']}")
    
    if 'zip_path' in result:
        print(f"\n📦 ZIP archive: {result['zip_path']}")
        if args.for_friend:
            print(f"   Ready to send to: {args.for_friend}")
    
    # Copy to data/ for local testing
    if args.copy_to_data:
        data_dir = script_dir.parent / 'data'
        import shutil
        src = os.path.join(output_dir, 'test_labels.csv')
        dst = data_dir / 'test_labels.csv'
        shutil.copy(src, dst)
        print(f"\n📋 Copied test_labels.csv to {dst}")
        print("   (For local testing only - do NOT commit!)")
    
    print("\n" + "=" * 60)
    print("⚠️  IMPORTANT REMINDERS:")
    print("  - test_labels.csv and valid_labels.csv are PRIVATE")
    print("  - Copy these to your PRIVATE repository (enigma-private)")
    print("  - Do NOT commit these to the public repository")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
