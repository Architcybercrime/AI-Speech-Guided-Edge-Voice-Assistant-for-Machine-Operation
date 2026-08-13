"""
augment.py — turn a small real dataset into a noise-robust training set.

Reads   data/raw/<label>/*.wav   (+ data/raw/_noise/*.wav)
Writes  data/augmented/<label>/*.wav

Usage:
    python tools/augment.py                 # default 6 variants per clip
    python tools/augment.py --per-clip 10
    python tools/augment.py --snr 0 5 10 15 20

Every output clip is exactly 1.0 s @ 16 kHz, peak-normalised to -3 dBFS,
which is what Edge Impulse expects.
"""

import argparse
import glob
import os
import random
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

SR = 16000
CLIP_LEN = SR  # 1 second

RAW = os.path.join("data", "raw")
OUT = os.path.join("data", "augmented")


def load(path):
    x, _ = librosa.load(path, sr=SR, mono=True)
    return x.astype(np.float32)


def fit_length(x, n=CLIP_LEN):
    if len(x) > n:
        s = (len(x) - n) // 2
        return x[s:s + n]
    if len(x) < n:
        pad = n - len(x)
        return np.pad(x, (pad // 2, pad - pad // 2))
    return x


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def mix_noise(sig, noise, snr_db):
    """Scale noise so that signal-to-noise ratio == snr_db."""
    if len(noise) < len(sig):
        noise = np.tile(noise, int(np.ceil(len(sig) / len(noise))))
    off = random.randint(0, len(noise) - len(sig))
    n = noise[off:off + len(sig)]
    target_noise_rms = rms(sig) / (10 ** (snr_db / 20.0))
    return sig + n * (target_noise_rms / rms(n))


def time_shift(x, max_ms=120):
    s = int(random.uniform(-max_ms, max_ms) * SR / 1000)
    return np.roll(x, s)


def speed_perturb(x, lo=0.9, hi=1.1):
    r = random.uniform(lo, hi)
    return fit_length(librosa.effects.time_stretch(x, rate=r))


def pitch_shift(x, semitones=2.0):
    s = random.uniform(-semitones, semitones)
    return librosa.effects.pitch_shift(x, sr=SR, n_steps=s)


def gain(x, lo_db=-8, hi_db=4):
    return x * (10 ** (random.uniform(lo_db, hi_db) / 20.0))


def normalize(x, peak_db=-3.0):
    p = float(np.max(np.abs(x)))
    if p < 1e-6:
        return x
    return x * ((10 ** (peak_db / 20.0)) / p)


def build_variant(x, noises, snr_choices):
    y = fit_length(x)
    if random.random() < 0.7:
        y = time_shift(y)
    if random.random() < 0.4:
        y = speed_perturb(y)
    if random.random() < 0.3:
        y = pitch_shift(y)
    if noises and random.random() < 0.85:
        y = mix_noise(y, random.choice(noises), random.choice(snr_choices))
    if random.random() < 0.5:
        y = gain(y)
    return normalize(fit_length(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", type=int, default=6,
                    help="augmented variants generated per source clip")
    ap.add_argument("--snr", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    noise_files = glob.glob(os.path.join(RAW, "_noise", "*.wav"))
    noises = [load(f) for f in noise_files]
    if not noises:
        print("WARNING: no files in data/raw/_noise/ — augmenting without noise "
              "mixing. Noise robustness will be poor. Record ambient audio first.")

    labels = sorted(d for d in os.listdir(RAW)
                    if os.path.isdir(os.path.join(RAW, d)) and d != "_noise")
    if not labels:
        print("No label folders found under data/raw/. Run record_dataset.py first.")
        return

    total = 0
    for label in labels:
        src = sorted(glob.glob(os.path.join(RAW, label, "*.wav")))
        dst = os.path.join(OUT, label)
        os.makedirs(dst, exist_ok=True)
        for path in tqdm(src, desc=f"{label:<10}", ncols=78):
            base = os.path.splitext(os.path.basename(path))[0]
            x = load(path)
            # keep one clean copy so the model also sees the easy case
            sf.write(os.path.join(dst, f"{base}_clean.wav"),
                     normalize(fit_length(x)), SR)
            total += 1
            for i in range(args.per_clip):
                y = build_variant(x, noises, args.snr)
                sf.write(os.path.join(dst, f"{base}_aug{i}.wav"), y, SR)
                total += 1

    # silence class, generated straight from ambient recordings
    if noises:
        dst = os.path.join(OUT, "_silence")
        os.makedirs(dst, exist_ok=True)
        n_sil = max(200, total // (len(labels) + 1))
        for i in range(n_sil):
            n = random.choice(noises)
            off = random.randint(0, max(0, len(n) - CLIP_LEN))
            seg = fit_length(n[off:off + CLIP_LEN])
            seg = seg * random.uniform(0.2, 1.0)
            sf.write(os.path.join(dst, f"sil_{i:04d}.wav"), normalize(seg), SR)
            total += 1

    print(f"\ndone — {total} clips written to {OUT}/")
    print("Next: zip data/augmented and upload to Edge Impulse "
          "(Data acquisition -> Upload data -> infer labels from folder names).")


if __name__ == "__main__":
    main()
