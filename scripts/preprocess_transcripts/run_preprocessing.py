"""
================================================================================
PREPROCESSING PIPELINE - Master Script
================================================================================

Runs the complete transcript preprocessing pipeline:
    Step 1: Extract transcripts from CHA files → transcripts_raw.csv
    Step 2: Clean CHAT annotations + count biomarkers → transcripts_clean.csv

This ONLY does preprocessing. Feature extraction is handled separately in:
    - feature_extraction/extract_linguistic_features.py (8 features)
    - feature_extraction/extract_acoustic_features.py (8 features) 
    - feature_extraction/extract_semantic_features.py (5 features)

Usage:
    python run_preprocessing.py -m "C:/FYP/data/metadata/metadata.csv" -o "C:/FYP/data/transcripts"
"""

import os
import sys
import argparse
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module


def main():
    parser = argparse.ArgumentParser(description='Run transcript preprocessing pipeline')
    parser.add_argument('--metadata', '-m', required=True, help='Path to metadata.csv')
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.metadata):
        print(f"Error: metadata not found: {args.metadata}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    
    print("=" * 70)
    print("TRANSCRIPT PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"  Metadata: {args.metadata}")
    print(f"  Output:   {args.output_dir}")
    print("=" * 70)
    
    # =========================================================
    # STEP 1: Extract raw transcripts from CHA files
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 1: Extracting Transcripts from CHA Files")
    print("=" * 70)
    
    # Import and run step 1
    from importlib.util import spec_from_file_location, module_from_spec
    step1_path = os.path.join(os.path.dirname(__file__), '01_extract_transcripts.py')
    spec = spec_from_file_location("step1", step1_path)
    step1 = module_from_spec(spec)
    spec.loader.exec_module(step1)
    
    raw_path = step1.build_raw_transcripts(args.metadata, args.output_dir)
    
    if not os.path.exists(raw_path):
        print("ERROR: Step 1 failed - transcripts_raw.csv not created")
        return
    
    # =========================================================
    # STEP 2: Clean transcripts + count biomarkers
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 2: Cleaning Transcripts + Counting Biomarkers")
    print("=" * 70)
    
    step2_path = os.path.join(os.path.dirname(__file__), '02_clean_transcripts.py')
    spec = spec_from_file_location("step2", step2_path)
    step2 = module_from_spec(spec)
    spec.loader.exec_module(step2)
    
    clean_path = step2.process_transcripts(raw_path, args.output_dir)
    
    elapsed = time.time() - start_time
    
    # =========================================================
    # Final Summary
    # =========================================================
    print("\n" + "=" * 70)
    print(f"✅ PREPROCESSING COMPLETE ({elapsed:.1f} seconds)")
    print("=" * 70)
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  1. {os.path.join(args.output_dir, 'transcripts_raw.csv')}")
    print(f"  2. {os.path.join(args.output_dir, 'transcripts_clean.csv')}")
    
    print(f"\n📊 BIOMARKER FEATURES READY (5 of 21):")
    print(f"  • filler_count")
    print(f"  • unfilled_pause_count")
    print(f"  • repetition_count")
    print(f"  • revision_count")
    print(f"  • fragment_count")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"  1. Run: python extract_linguistic_features.py  (8 features)")
    print(f"  2. Run: python extract_acoustic_features.py   (8 features)")
    print(f"  3. Run: python extract_semantic_features.py   (5 features)")
    print(f"  4. Merge all features into final matrix")
    print(f"  5. Train baseline classifiers")


if __name__ == '__main__':
    main()
