"""
record_dataset.py — collect keyword clips using the laptop microphone.
No project hardware required.

Usage:
    python tools/record_dataset.py --speaker archit
    python tools/record_dataset.py --speaker riya --reps 20
    python tools/record_dataset.py --speaker archit --noise      # ambient only

Controls: press ENTER to record one clip, 's' + ENTER to skip a word,
          'q' + ENTER to quit early. Files land in data/raw/<label>/.
"""

import argparse
import os
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000
CLIP_SEC = 1.0
NOISE_SEC = 60.0

COMMANDS = ["start", "halt", "pause", "resume",
            "faster", "slower", "confirm", "cancel"]

# Phonetically varied fillers. Include near-rhymes of the real commands so the
# model learns to reject them instead of snapping to the nearest keyword.
UNKNOWN = ["yes", "no", "hello", "okay", "restart", "smart", "salt",
           "faster machine", "one", "two", "three", "four", "five",
           "chalo", "band karo", "ruko", "theek hai", "kya hua"]


def outdir(label):
    d = os.path.join("data", "raw", label)
    os.makedirs(d, exist_ok=True)
    return d


def next_index(d, speaker):
    n = len([f for f in os.listdir(d) if f.startswith(speaker + "_")])
    return n + 1


def record(seconds):
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def peak_ok(x):
    p = float(np.max(np.abs(x)))
    if p < 0.02:
        return False, f"too quiet (peak {p:.3f}) — move closer"
    if p > 0.99:
        return False, f"clipping (peak {p:.3f}) — move back"
    return True, f"peak {p:.3f}"


def capture_word(label, speaker, reps, distance_hint):
    d = outdir(label)
    idx = next_index(d, speaker)
    got = 0
    print(f"\n=== {label.upper()}  ({reps} reps) ===")
    print(f"    {distance_hint}")
    while got < reps:
        cmd = input(f"  [{got+1}/{reps}] ENTER=record  s=skip word  q=quit > ").strip().lower()
        if cmd == "q":
            return "quit"
        if cmd == "s":
            return "skip"
        print("  recording...", end="", flush=True)
        x = record(CLIP_SEC)
        ok, msg = peak_ok(x)
        print(f" {msg}")
        if not ok:
            print("  -> rejected, retrying")
            continue
        path = os.path.join(d, f"{speaker}_{idx:03d}.wav")
        sf.write(path, x, SR)
        idx += 1
        got += 1
    return "done"


def capture_noise(speaker):
    d = outdir("_noise")
    idx = next_index(d, speaker)
    print(f"\n=== AMBIENT NOISE ({int(NOISE_SEC)} s) ===")
    print("    Leave the room noisy: fans, chatter, traffic, keyboard.")
    print("    Do NOT speak any command words.")
    input("  ENTER to start > ")
    print("  recording...", end="", flush=True)
    x = record(NOISE_SEC)
    path = os.path.join(d, f"{speaker}_{idx:03d}.wav")
    sf.write(path, x, SR)
    print(f" saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True, help="short id, e.g. archit")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--unknown-reps", type=int, default=2)
    ap.add_argument("--noise", action="store_true", help="record ambient only")
    args = ap.parse_args()

    print(f"device: {sd.query_devices(kind='input')['name']}  @ {SR} Hz")

    if args.noise:
        capture_noise(args.speaker)
        return

    hints = [
        "Speak normally, about 1 arm's length from the mic.",
        "Now step back ~2 m and speak louder.",
        "Now speak quietly / tired, close to the mic.",
    ]

    for i, w in enumerate(COMMANDS):
        r = capture_word(w, args.speaker, args.reps, hints[i % len(hints)])
        if r == "quit":
            sys.exit(0)

    print("\n=== UNKNOWN CLASS (rejection training) ===")
    for w in UNKNOWN:
        d = outdir("_unknown")
        idx = next_index(d, args.speaker)
        for k in range(args.unknown_reps):
            cmd = input(f"  say \"{w}\" [{k+1}/{args.unknown_reps}] ENTER  (s=skip) > ").strip().lower()
            if cmd == "s":
                break
            print("  recording...", end="", flush=True)
            x = record(CLIP_SEC)
            ok, msg = peak_ok(x)
            print(f" {msg}")
            if not ok:
                continue
            sf.write(os.path.join(d, f"{args.speaker}_{idx:03d}.wav"), x, SR)
            idx += 1

    print("\nNow record ambient noise:")
    print(f"  python tools/record_dataset.py --speaker {args.speaker} --noise")


if __name__ == "__main__":
    main()
