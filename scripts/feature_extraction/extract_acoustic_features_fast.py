"""
FAST Acoustic Feature Extraction using Parselmouth (Praat)
Optimized version - ~20-50x faster than librosa-based approach

Extracts 8 acoustic features:
1. pause_rate - Number of pauses per second
2. mean_pause_duration - Average pause length in seconds
3. speech_time_ratio - Proportion of time spent speaking
4. pitch_range - Difference between max and min F0
5. pitch_variability - Standard deviation of F0
6. intensity_range - Dynamic range in dB
7. jitter - Pitch perturbation (voice quality)
8. shimmer - Amplitude perturbation (voice quality)
"""

import os
import argparse
import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def extract_features_single(wav_path: str) -> dict:
    """Extract all 8 acoustic features from a single WAV file using Parselmouth."""
    try:
        # Load audio with Parselmouth (handles MP3-in-WAV correctly)
        snd = parselmouth.Sound(wav_path)
        duration = snd.get_total_duration()
        
        if duration < 0.5:  # Too short
            return None
        
        # === PITCH FEATURES ===
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=50, pitch_ceiling=300)
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced
        
        if len(pitch_values) > 0:
            pitch_range = float(np.max(pitch_values) - np.min(pitch_values))
            pitch_variability = float(np.std(pitch_values))
        else:
            pitch_range = 0.0
            pitch_variability = 0.0
        
        # === INTENSITY FEATURES ===
        intensity = snd.to_intensity(minimum_pitch=50)
        intensity_values = intensity.values[0]
        intensity_values = intensity_values[~np.isnan(intensity_values)]
        
        if len(intensity_values) > 0:
            intensity_range = float(np.max(intensity_values) - np.min(intensity_values))
        else:
            intensity_range = 0.0
        
        # === PAUSE DETECTION ===
        # Use intensity to detect pauses (low intensity = pause)
        intensity_threshold = np.percentile(intensity_values, 25) if len(intensity_values) > 0 else 0
        time_step = intensity.time_step
        
        is_pause = intensity_values < intensity_threshold
        
        # Count pause segments
        pause_starts = np.where(np.diff(is_pause.astype(int)) == 1)[0]
        pause_ends = np.where(np.diff(is_pause.astype(int)) == -1)[0]
        
        # Handle edge cases
        if is_pause[0]:
            pause_starts = np.insert(pause_starts, 0, 0)
        if is_pause[-1]:
            pause_ends = np.append(pause_ends, len(is_pause) - 1)
        
        # Calculate pause metrics
        min_len = min(len(pause_starts), len(pause_ends))
        if min_len > 0:
            pause_durations = (pause_ends[:min_len] - pause_starts[:min_len]) * time_step
            pause_durations = pause_durations[pause_durations > 0.1]  # Only pauses > 100ms
            
            num_pauses = len(pause_durations)
            pause_rate = float(num_pauses / duration)
            mean_pause_duration = float(np.mean(pause_durations)) if num_pauses > 0 else 0.0
            total_pause_time = float(np.sum(pause_durations))
            speech_time_ratio = float(1 - (total_pause_time / duration))
        else:
            pause_rate = 0.0
            mean_pause_duration = 0.0
            speech_time_ratio = 1.0
        
        # Clamp speech_time_ratio
        speech_time_ratio = max(0.0, min(1.0, speech_time_ratio))
        
        # === VOICE QUALITY (Jitter & Shimmer) ===
        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", 50, 300)
            
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            jitter = float(jitter) if not np.isnan(jitter) else 0.0
            shimmer = float(shimmer) if not np.isnan(shimmer) else 0.0
        except:
            jitter = 0.0
            shimmer = 0.0
        
        return {
            'pause_rate': round(pause_rate, 4),
            'mean_pause_duration': round(mean_pause_duration, 4),
            'speech_time_ratio': round(speech_time_ratio, 4),
            'pitch_range': round(pitch_range, 2),
            'pitch_variability': round(pitch_variability, 2),
            'intensity_range': round(intensity_range, 2),
            'jitter': round(jitter, 6),
            'shimmer': round(shimmer, 6),
        }
        
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None


def process_row(args):
    """Process a single row - for parallel execution."""
    idx, row, audio_dir = args
    
    # Build WAV path
    wav_filename = f"{int(row['participant_id']):03d}-{int(row['session'])}.wav"
    wav_path = os.path.join(audio_dir, row['folder_group'], row['task_folder'], wav_filename)
    
    if not os.path.exists(wav_path):
        return idx, None
    
    features = extract_features_single(wav_path)
    
    if features is None:
        return idx, None
    
    # Add identifiers
    features['participant_id'] = row['participant_id']
    features['session'] = row['session']
    features['task_folder'] = row['task_folder']
    features['diagnosis'] = row['diagnosis']
    
    return idx, features


def extract_acoustic_features(audio_dir: str, metadata_path: str, output_dir: str, n_workers: int = 4):
    """Extract acoustic features from all WAV files."""
    
    print("=" * 70)
    print("FAST ACOUSTIC FEATURE EXTRACTION (Parselmouth)")
    print("=" * 70)
    print(f"Audio dir:  {audio_dir}")
    print(f"Metadata:   {metadata_path}")
    print(f"Output:     {output_dir}")
    print(f"Workers:    {n_workers}")
    print("=" * 70)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    print(f"Found {len(metadata)} records in metadata")
    
    # Prepare arguments for parallel processing
    args_list = [(idx, row, audio_dir) for idx, row in metadata.iterrows()]
    
    # Process in parallel
    results = []
    failed = 0
    
    print(f"\nExtracting 8 acoustic features from {len(metadata)} files...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_row, args): args[0] for args in args_list}
        
        with tqdm(total=len(futures), desc="Processing", unit="file") as pbar:
            for future in as_completed(futures):
                idx, features = future.result()
                if features:
                    results.append(features)
                else:
                    failed += 1
                pbar.update(1)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['participant_id', 'session', 'task_folder', 'diagnosis',
            'pause_rate', 'mean_pause_duration', 'speech_time_ratio',
            'pitch_range', 'pitch_variability', 'intensity_range',
            'jitter', 'shimmer']
    df = df[cols]
    
    # Save
    output_path = os.path.join(output_dir, 'features_acoustic.csv')
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Successful: {len(df)}")
    print(f"Failed:     {failed}")
    print(f"Output:     {output_path}")
    print("\nFeature statistics:")
    print(df[['pause_rate', 'mean_pause_duration', 'speech_time_ratio',
              'pitch_range', 'pitch_variability', 'intensity_range',
              'jitter', 'shimmer']].describe().round(4))
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast acoustic feature extraction")
    parser.add_argument("--audio_dir", type=str, default="D:/FYP/data/audio")
    parser.add_argument("--metadata", type=str, default="D:/FYP/data/metadata/metadata.csv")
    parser.add_argument("--output", type=str, default="D:/FYP/data/features")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    extract_acoustic_features(args.audio_dir, args.metadata, args.output, args.workers)
