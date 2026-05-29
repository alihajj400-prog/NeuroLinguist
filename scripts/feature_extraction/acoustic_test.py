import time
import librosa
import parselmouth

wav_path = "D:/FYP/data/audio/Control/cookie/002-0.wav"  # pick any file

# Test 1: Loading
t0 = time.time()
y, sr = librosa.load(wav_path, sr=None)  # sr=None = no resampling
print(f"Load (no resample): {time.time()-t0:.2f}s, len={len(y)/sr:.1f}s")

# Test 2: Load with resample (default)
t0 = time.time()
y2, sr2 = librosa.load(wav_path)  # default resamples to 22050
print(f"Load (resample): {time.time()-t0:.2f}s")

# Test 3: Pitch with pyin (SLOW)
t0 = time.time()
f0, _, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
print(f"pyin pitch: {time.time()-t0:.2f}s")

# Test 4: Pitch with parselmouth (FAST)
t0 = time.time()
snd = parselmouth.Sound(wav_path)
pitch = snd.to_pitch()
print(f"parselmouth pitch: {time.time()-t0:.2f}s")