"""
features.py — MFCC front-end shared by training and live inference.

Both paths MUST use this module. The single most common cause of
"97% in training, 60% on device" is a mismatch between how features
were computed at training time and at inference time.
"""

import numpy as np
import librosa

SR = 16000
CLIP_LEN = SR          # 1 second
N_MFCC = 20            # cepstral coefficients kept
N_MELS = 40            # mel filter bank size
N_FFT = 480            # 30 ms window
HOP = 320              # 20 ms hop  -> 51 frames per second
N_FRAMES = 1 + CLIP_LEN // HOP


def fit_length(x, n=CLIP_LEN):
    if len(x) > n:
        s = (len(x) - n) // 2
        return x[s:s + n]
    if len(x) < n:
        pad = n - len(x)
        return np.pad(x, (pad // 2, pad - pad // 2))
    return x


def normalize(x, peak=0.7):
    p = float(np.max(np.abs(x)))
    return x if p < 1e-6 else x * (peak / p)


def mfcc(wave, sr=SR):
    """float32 waveform -> (N_MFCC, N_FRAMES) float32 feature map."""
    x = normalize(fit_length(np.asarray(wave, dtype=np.float32)))
    m = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS,
                             n_fft=N_FFT, hop_length=HOP)
    # per-utterance normalisation: removes channel and loudness differences
    m = (m - m.mean()) / (m.std() + 1e-6)
    if m.shape[1] < N_FRAMES:
        m = np.pad(m, ((0, 0), (0, N_FRAMES - m.shape[1])))
    return m[:, :N_FRAMES].astype(np.float32)


def load_mfcc(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return mfcc(y)


INPUT_SHAPE = (N_MFCC, N_FRAMES, 1)
