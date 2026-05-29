import os
import subprocess
from pathlib import Path

RAW_PATH = Path(r"C:\FYP\data\raw")
PROCESSED_PATH = Path(r"C:\FYP\data\audio")

GROUPS = ["Control", "Dementia"]
TASKS = ["cookie", "fluency", "recall", "sentence"]

SAMPLE_RATE = 16000  # 16kHz


def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to 16kHz, 16-bit, mono WAV using FFmpeg"""
    command = [
        "ffmpeg",
        "-i", str(mp3_path),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-y",
        str(wav_path)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0


def main():
    print("=" * 50)
    print("AUDIO PREPROCESSING: MP3 → WAV")
    print("(16kHz, 16-bit, mono)")
    print("Only files with matching transcripts")
    print("=" * 50)
    
    total_files = 0
    converted = 0
    failed = 0
    skipped = 0
    no_transcript = 0
    
    for group in GROUPS:
        for task in TASKS:
            raw_folder = RAW_PATH / group / task
            processed_folder = PROCESSED_PATH / group / task
            
            # Create processed folder
            processed_folder.mkdir(parents=True, exist_ok=True)
            
            if not raw_folder.exists():
                print(f"\n⚠ Folder not found: {raw_folder}")
                continue
            
            mp3_files = list(raw_folder.glob("*.mp3"))
            cha_files = set(f.stem for f in raw_folder.glob("*.cha"))
            
            print(f"\n📁 {group}/{task}")
            print(f"    MP3 files: {len(mp3_files)}")
            print(f"    CHA files: {len(cha_files)}")
            
            for mp3_file in mp3_files:
                total_files += 1
                
                # Check if matching transcript exists
                if mp3_file.stem not in cha_files:
                    no_transcript += 1
                    print(f"    ⚠ No transcript for: {mp3_file.name}")
                    continue
                
                wav_file = processed_folder / mp3_file.name.replace(".mp3", ".wav")
                
                # Skip if already converted
                if wav_file.exists() and wav_file.stat().st_size > 1000:
                    skipped += 1
                    continue
                
                print(f"    Converting: {mp3_file.name}...", end=" ", flush=True)
                
                if convert_mp3_to_wav(mp3_file, wav_file):
                    print("OK")
                    converted += 1
                else:
                    print("FAILED")
                    failed += 1
    
    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"Total MP3 files: {total_files}")
    print(f"✅ Converted: {converted}")
    print(f"⏭  Skipped (already exists): {skipped}")
    print(f"⚠  No matching transcript: {no_transcript}")
    print(f"❌ Failed: {failed}")


if __name__ == "__main__":
    main()