"""
Metadata Extraction Script for DementiaBank Pitt Corpus
========================================================
Usage:
    python build_metadata.py -r "C:/FYP/data/raw" -p "C:/FYP/data/audio" -o "C:/FYP/data/metadata/metadata.csv"
"""

import os
import re
import csv
import argparse
from pathlib import Path

# Map detailed task names to folder names
TASK_TO_FOLDER = {
    # Cookie
    'cookie': 'cookie',
    # Fluency variants
    'animals': 'fluency',
    'foods': 'fluency',
    'fluency': 'fluency',
    # Recall variants
    'george_immediate': 'recall',
    'george_delayed': 'recall',
    'george_immmediate': 'recall',  # typo in data
    'george_unknown': 'recall',
    'bill_immediate': 'recall',
    'bill_delayed': 'recall',
    'story_immediate': 'recall',
    'unknown_immediate': 'recall',
    'recall': 'recall',
    # Sentence
    'sentence': 'sentence',
}


def parse_cha_file(cha_path):
    metadata = {
        'age': None,
        'sex': None,
        'diagnosis': None,
        'education': None,
        'task_detail': None,
    }
    
    try:
        try:
            with open(cha_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(cha_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('@ID:') and '|PAR|' in line:
                parts = line.split('|')
                if len(parts) >= 9:
                    age_str = parts[3].replace(';', '').strip()
                    if age_str.isdigit():
                        metadata['age'] = int(age_str)
                    metadata['sex'] = parts[4].strip() if parts[4].strip() else None
                    metadata['diagnosis'] = parts[5].strip() if parts[5].strip() else None
                    edu_str = parts[8].strip() if len(parts) > 8 else ''
                    if edu_str.isdigit():
                        metadata['education'] = int(edu_str)
            
            if line.startswith('@G:'):
                metadata['task_detail'] = line.replace('@G:', '').strip().lower()
            
            if line.startswith('*PAR:') or line.startswith('*INV:'):
                break
                
    except Exception as e:
        print(f"Error parsing {cha_path}: {e}")
    
    return metadata


def extract_participant_session(filename):
    stem = Path(filename).stem
    match = re.match(r'^(\d+)-(\d+)$', stem)
    if match:
        return match.group(1), match.group(2)
    numbers = re.findall(r'\d+', stem)
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    elif len(numbers) == 1:
        return numbers[0], '0'
    return stem, '0'


def get_folder_group(path):
    path_lower = path.lower()
    if 'control' in path_lower:
        return 'Control'
    elif 'dementia' in path_lower:
        return 'Dementia'
    return 'Unknown'


def get_task_from_path(path):
    path_lower = path.lower()
    for task_name in ['cookie', 'fluency', 'recall', 'sentence']:
        if task_name in path_lower:
            return task_name
    return 'unknown'


def get_task_folder(task_detail):
    """Map detailed task name to folder name."""
    if not task_detail:
        return None
    task_lower = task_detail.lower().strip()
    return TASK_TO_FOLDER.get(task_lower, None)


def find_wav_file(participant_id, session, folder_group, task_folder, processed_dir):
    if not processed_dir or not task_folder:
        return None
    
    wav_filename = f"{participant_id}-{session}.wav"
    
    # Try exact path
    wav_path = os.path.join(processed_dir, folder_group, task_folder, wav_filename)
    if os.path.exists(wav_path):
        return wav_path
    
    # Try case variations
    for g in [folder_group, folder_group.lower(), folder_group.capitalize()]:
        for t in [task_folder, task_folder.lower(), task_folder.capitalize()]:
            test_path = os.path.join(processed_dir, g, t, wav_filename)
            if os.path.exists(test_path):
                return test_path
    
    return None


def build_metadata(raw_dir, processed_dir, output_path):
    records = []
    cha_files_found = 0
    unmapped_tasks = set()
    
    for root, dirs, files in os.walk(raw_dir):
        for filename in files:
            if not filename.endswith('.cha'):
                continue
            
            cha_files_found += 1
            cha_path = os.path.join(root, filename)
            
            participant_id, session = extract_participant_session(filename)
            meta = parse_cha_file(cha_path)
            
            folder_group = get_folder_group(cha_path)
            
            # Get task_detail from CHA, fallback to path
            task_detail = meta['task_detail'] if meta['task_detail'] else get_task_from_path(cha_path)
            
            # Map to folder name
            task_folder = get_task_folder(task_detail)
            if not task_folder:
                task_folder = get_task_from_path(cha_path)
                if task_detail:
                    unmapped_tasks.add(task_detail)
            
            wav_path = find_wav_file(participant_id, session, folder_group, task_folder, processed_dir)
            
            record = {
                'participant_id': participant_id,
                'session': session,
                'folder_group': folder_group,
                'diagnosis': meta['diagnosis'],
                'task_folder': task_folder,
                'task_detail': task_detail,
                'age': meta['age'],
                'sex': meta['sex'],
                'education': meta['education'],
                'cha_path': cha_path,
                'wav_path': wav_path if wav_path else '',
                'has_wav': 'yes' if wav_path else 'no'
            }
            records.append(record)
    
    print(f"Found {cha_files_found} CHA files")
    print(f"Processed {len(records)} records")
    
    if unmapped_tasks:
        print(f"\n⚠️  Unmapped tasks (using folder path): {unmapped_tasks}")
    
    records.sort(key=lambda x: (x['participant_id'], x['session']))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    print("\nBy Folder Group:")
    for group in ['Control', 'Dementia']:
        count = sum(1 for r in records if r['folder_group'] == group)
        print(f"  {group}: {count}")
    
    print("\nBy Diagnosis:")
    diagnoses = {}
    for r in records:
        d = r['diagnosis'] or 'Unknown'
        diagnoses[d] = diagnoses.get(d, 0) + 1
    for d, count in sorted(diagnoses.items(), key=lambda x: -x[1]):
        print(f"  {d}: {count}")
    
    print("\nBy Task Folder:")
    task_folders = {}
    for r in records:
        t = r['task_folder'] or 'Unknown'
        task_folders[t] = task_folders.get(t, 0) + 1
    for t, count in sorted(task_folders.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")
    
    print("\nBy Task Detail:")
    task_details = {}
    for r in records:
        t = r['task_detail'] or 'Unknown'
        task_details[t] = task_details.get(t, 0) + 1
    for t, count in sorted(task_details.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")
    
    wav_count = sum(1 for r in records if r['has_wav'] == 'yes')
    print(f"\nRecords with WAV files: {wav_count}/{len(records)}")
    
    print("\nWAV files by folder group:")
    for group in ['Control', 'Dementia']:
        count = sum(1 for r in records if r['folder_group'] == group and r['has_wav'] == 'yes')
        total = sum(1 for r in records if r['folder_group'] == group)
        print(f"  {group}: {count}/{total}")
    
    print("\nWAV files by task folder:")
    for task in ['cookie', 'fluency', 'recall', 'sentence']:
        count = sum(1 for r in records if r['task_folder'] == task and r['has_wav'] == 'yes')
        total = sum(1 for r in records if r['task_folder'] == task)
        print(f"  {task}: {count}/{total}")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = [
            'participant_id', 'session', 'folder_group', 'diagnosis', 
            'task_folder', 'task_detail',
            'age', 'sex', 'education', 'cha_path', 'wav_path', 'has_wav'
        ]
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"\nMetadata saved to: {output_path}")
    
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', '-r', required=True)
    parser.add_argument('--processed_dir', '-p', required=True)
    parser.add_argument('--output', '-o', default='metadata.csv')
    args = parser.parse_args()
    
    if not os.path.exists(args.raw_dir):
        print(f"Error: raw_dir not found: {args.raw_dir}")
        return
    if not os.path.exists(args.processed_dir):
        print(f"Error: processed_dir not found: {args.processed_dir}")
        return
    
    build_metadata(args.raw_dir, args.processed_dir, args.output)


if __name__ == '__main__':
    main()