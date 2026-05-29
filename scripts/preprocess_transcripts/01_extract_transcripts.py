"""
================================================================================
STEP 1: Extract Participant Speech from CHA Files
================================================================================

Reads metadata.csv, extracts *PAR lines from each CHA file.
Outputs: transcripts_raw.csv

Key improvements:
- Robust path resolution (handles relative/absolute paths)
- Preserves ALL CHAT markers for later biomarker counting
- Handles multi-line utterances correctly
- Better error reporting

Usage:
    python 01_extract_transcripts.py -m "C:/FYP/Pitt/metadata/metadata.csv" -o "C:/FYP/Pitt/metadata"
"""

import os
import re
import csv
import argparse
from pathlib import Path


def resolve_cha_path(cha_path, metadata_dir):
    """
    Resolve CHA file path - handles relative paths from metadata location.
    """
    # If absolute and exists, use it
    if os.path.isabs(cha_path) and os.path.exists(cha_path):
        return cha_path
    
    # Try as-is
    if os.path.exists(cha_path):
        return cha_path
    
    # Try relative to metadata directory
    relative_path = os.path.join(metadata_dir, cha_path)
    if os.path.exists(relative_path):
        return relative_path
    
    # Try going up one level from metadata (common structure)
    parent_relative = os.path.join(os.path.dirname(metadata_dir), cha_path)
    if os.path.exists(parent_relative):
        return parent_relative
    
    # Try with normalized path separators
    normalized = cha_path.replace('\\', '/').replace('/', os.sep)
    if os.path.exists(normalized):
        return normalized
    
    return None


def extract_par_lines(cha_path):
    """
    Extract ALL *PAR: utterances from a CHA file.
    
    CRITICAL: Preserves ALL CHAT markers including:
    - Filled pauses: &-uh, &-um, &-hm, &-mm, &-er, &-ah
    - Unfilled pauses: (.), (..), (...)
    - Repetitions: [/]
    - Revisions: [//], [///]
    - Fragments: &+word
    - Timing markers: •0_1234• or \x150_1234\x15
    
    Returns: list of raw utterance strings (one per *PAR line)
    """
    utterances = []
    current_utterance = None
    
    # Try multiple encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    lines = None
    
    for encoding in encodings:
        try:
            with open(cha_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if lines is None:
        print(f"    ⚠️ Could not read file with any encoding: {cha_path}")
        return []
    
    for line in lines:
        # New PAR utterance starts
        if line.startswith('*PAR:'):
            # Save previous utterance if exists
            if current_utterance is not None:
                utterances.append(current_utterance.strip())
            # Start new utterance (remove *PAR: prefix)
            current_utterance = line[5:].strip()  # Skip '*PAR:'
        
        # Continuation line (tab-indented, part of current PAR utterance)
        elif current_utterance is not None and line.startswith('\t'):
            stripped = line.strip()
            # Skip dependent tiers (%mor, %gra, etc.) and headers (@)
            if not stripped.startswith('%') and not stripped.startswith('@'):
                current_utterance += ' ' + stripped
        
        # Hit a different speaker (*INV, *CHI, etc.), dependent tier, or header
        elif current_utterance is not None and (
            line.startswith('*') or 
            line.startswith('%') or 
            line.startswith('@')
        ):
            utterances.append(current_utterance.strip())
            current_utterance = None
    
    # Don't forget last utterance
    if current_utterance is not None:
        utterances.append(current_utterance.strip())
    
    return utterances


def build_raw_transcripts(metadata_path, output_dir):
    """
    Read metadata.csv, extract PAR lines from each CHA file.
    Output: transcripts_raw.csv
    """
    metadata_dir = os.path.dirname(os.path.abspath(metadata_path))
    output_path = os.path.join(output_dir, 'transcripts_raw.csv')
    
    records = []
    skipped = 0
    found = 0
    
    # Read metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        metadata_rows = list(reader)
    
    print(f"Processing {len(metadata_rows)} records from metadata...")
    print(f"  Metadata directory: {metadata_dir}")
    
    # Show first few CHA paths for debugging
    if metadata_rows:
        print(f"\n  Sample CHA paths from metadata:")
        for i, row in enumerate(metadata_rows[:3]):
            print(f"    {i+1}. {row.get('cha_path', 'N/A')}")
    
    for i, row in enumerate(metadata_rows):
        cha_path_raw = row.get('cha_path', '')
        
        # Resolve the actual path
        cha_path = resolve_cha_path(cha_path_raw, metadata_dir)
        
        if cha_path is None:
            if skipped < 5:  # Only show first 5 warnings
                print(f"  ⚠️ CHA not found: {cha_path_raw}")
            skipped += 1
            continue
        
        found += 1
        
        # Extract PAR utterances
        utterances = extract_par_lines(cha_path)
        
        # Keep utterances separate with delimiter for later processing
        # Use ||| as utterance boundary marker
        raw_text = ' ||| '.join(utterances)
        utterance_count = len(utterances)
        
        record = {
            'participant_id': row.get('participant_id', ''),
            'session': row.get('session', ''),
            'folder_group': row.get('folder_group', ''),
            'diagnosis': row.get('diagnosis', ''),
            'task_folder': row.get('task_folder', ''),
            'task_detail': row.get('task_detail', ''),
            'cha_path': cha_path,  # Store resolved path
            'raw_text': raw_text,
            'utterance_count': utterance_count,
        }
        records.append(record)
        
        # Progress
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(metadata_rows)} ({found} found, {skipped} skipped)")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Records processed: {len(records)}")
    print(f"  CHA files found: {found}")
    print(f"  CHA files not found: {skipped}")
    
    empty = sum(1 for r in records if not r['raw_text'].strip())
    print(f"  Empty transcripts: {empty}")
    
    # Stats
    non_empty = [r for r in records if r['raw_text'].strip()]
    if non_empty:
        word_counts = [len(r['raw_text'].split()) for r in non_empty]
        print(f"\n  Word count stats (raw):")
        print(f"    Average: {sum(word_counts)/len(word_counts):.1f}")
        print(f"    Min: {min(word_counts)}")
        print(f"    Max: {max(word_counts)}")
        
        # Show sample of raw text to verify markers are preserved
        print(f"\n  Sample raw text (first record with content):")
        sample = non_empty[0]['raw_text'][:300]
        print(f"    {sample}...")
    
    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        'participant_id', 'session', 'folder_group', 'diagnosis',
        'task_folder', 'task_detail', 'cha_path', 'raw_text', 'utterance_count'
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    print(f"\n✅ Saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Extract PAR transcripts from CHA files')
    parser.add_argument('--metadata', '-m', required=True, help='Path to metadata.csv')
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.metadata):
        print(f"Error: metadata not found: {args.metadata}")
        return
    
    build_raw_transcripts(args.metadata, args.output_dir)


if __name__ == '__main__':
    main()
