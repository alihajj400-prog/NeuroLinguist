"""
Fix invalid WAV files by converting them to proper WAV format using ffmpeg.
Skips truly corrupted files that can't be converted.
"""

import os
import subprocess
import shutil

audio_dir = r'D:\FYP\data\audio'
fixed = 0
failed = 0
already_valid = 0
failed_files = []

print("=" * 60)
print("FIXING INVALID WAV FILES")
print("=" * 60)

for root, dirs, files in os.walk(audio_dir):
    for f in files:
        if f.endswith('.wav'):
            path = os.path.join(root, f)
            
            # Check if valid WAV (starts with RIFF)
            try:
                with open(path, 'rb') as file:
                    header = file.read(4)
            except:
                failed += 1
                failed_files.append(path)
                continue
            
            if header != b'RIFF':
                temp = path + '.tmp.wav'
                
                try:
                    # Convert to proper WAV using ffmpeg
                    result = subprocess.run(
                        ['ffmpeg', '-y', '-i', path, '-acodec', 'pcm_s16le', '-ar', '16000', temp],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    # Check if conversion succeeded
                    if os.path.exists(temp) and os.path.getsize(temp) > 1000:
                        # Verify the new file has RIFF header
                        with open(temp, 'rb') as check:
                            new_header = check.read(4)
                        
                        if new_header == b'RIFF':
                            os.remove(path)
                            shutil.move(temp, path)
                            fixed += 1
                            print(f"Fixed: {f}")
                        else:
                            failed += 1
                            failed_files.append(path)
                            print(f"FAILED (bad output): {f}")
                            os.remove(temp)
                    else:
                        failed += 1
                        failed_files.append(path)
                        print(f"FAILED (no output): {f}")
                        if os.path.exists(temp):
                            os.remove(temp)
                            
                except subprocess.TimeoutExpired:
                    failed += 1
                    failed_files.append(path)
                    print(f"FAILED (timeout): {f}")
                    if os.path.exists(temp):
                        os.remove(temp)
                except Exception as e:
                    failed += 1
                    failed_files.append(path)
                    print(f"FAILED ({type(e).__name__}): {f}")
                    if os.path.exists(temp):
                        os.remove(temp)
            else:
                already_valid += 1

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Already valid: {already_valid}")
print(f"Fixed:         {fixed}")
print(f"Failed:        {failed}")
print("=" * 60)

if failed_files:
    print(f"\nFailed files saved to: D:\\FYP\\failed_wav_files.txt")
    with open(r'D:\FYP\failed_wav_files.txt', 'w') as f:
        for path in failed_files:
            f.write(path + '\n')
