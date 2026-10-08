"""
record_dataset.py — collect keyword clips using the laptop microphone.
No project hardware required.

Usage:
    python tools/record_dataset.py --speaker archit
    python tools/record_dataset.py --speaker riya --reps 20
    python tools/record_dataset.py --speaker archit --noise      # ambient only
    python tools/record_dataset.py --speaker archit --only hey_machina

Controls: ENTER records one clip, 's' skips the current word, 'q' quits.
Files land in data/raw/<label>/<speaker>_<nnn>.wav

Speaker count matters more than clip count. Ten people giving twenty reps
each beats one person giving two hundred.
"""

import argparse
import os
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000
CLIP_SEC = 1.2          # wake word needs a little more room than a single word
NOISE_SEC = 60.0

WAKE = 'hey_machina'
COMMANDS = ['start', 'halt', 'pause', 'resume',
            'faster', 'slower', 'confirm', 'cancel']

# Phonetically varied fillers, including near-rhymes of the real commands so
# the model learns to reject them instead of snapping to the nearest keyword.
UNKNOWN = ['yes', 'no', 'hello', 'okay', 'restart', 'smart', 'salt', 'halted',
           'hey machine', 'machina', 'faster machine', 'cancel that',
           'one', 'two', 'three', 'four', 'five',
           'chalo', 'band karo', 'ruko', 'theek hai', 'kya hua', 'suno']


def outdir(label):
    d = os.path.join('data', 'raw', label)
    os.makedirs(d, exist_ok=True)
    return d


def next_index(d, speaker):
    return len([f for f in os.listdir(d) if f.startswith(speaker + '_')]) + 1


def record(seconds):
    a = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    return a.reshape(-1)


def peak_ok(x):
    p = float(np.max(np.abs(x)))
    if p < 0.02:
        return False, f'too quiet (peak {p:.3f}) - move closer'
    if p > 0.99:
        return False, f'clipping (peak {p:.3f}) - move back'
    return True, f'peak {p:.3f}'


def capture(label, speaker, reps, hint, prompt=None):
    d = outdir(label)
    idx = next_index(d, speaker)
    got = 0
    print(f'\n=== {(prompt or label).upper()}   ({reps} reps) ===')
    print(f'    {hint}')
    while got < reps:
        cmd = input(f'  [{got+1}/{reps}] ENTER=record  s=skip  q=quit > ').strip().lower()
        if cmd == 'q':
            return 'quit'
        if cmd == 's':
            return 'skip'
        print('  recording...', end='', flush=True)
        x = record(CLIP_SEC)
        ok, msg = peak_ok(x)
        print(f' {msg}')
        if not ok:
            print('  -> rejected, retrying')
            continue
        sf.write(os.path.join(d, f'{speaker}_{idx:03d}.wav'), x, SR)
        idx += 1
        got += 1
    return 'done'


def capture_noise(speaker):
    d = outdir('_noise')
    idx = next_index(d, speaker)
    print(f'\n=== AMBIENT NOISE ({int(NOISE_SEC)} s) ===')
    print('    Leave the room noisy: fans, chatter, traffic, keyboard.')
    print('    Do NOT speak any command word. This file is what the false-accept')
    print('    rate is measured against, so record several minutes in total.')
    input('  ENTER to start > ')
    print('  recording...', end='', flush=True)
    x = record(NOISE_SEC)
    p = os.path.join(d, f'{speaker}_{idx:03d}.wav')
    sf.write(p, x, SR)
    print(f' saved {p}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speaker', required=True, help='short id, e.g. archit')
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--wake-reps', type=int, default=30)
    ap.add_argument('--unknown-reps', type=int, default=2)
    ap.add_argument('--noise', action='store_true')
    ap.add_argument('--only', help='record a single label only')
    args = ap.parse_args()

    print(f"device: {sd.query_devices(kind='input')['name']}  @ {SR} Hz")

    if args.noise:
        capture_noise(args.speaker)
        return

    hints = ['Speak normally, about an arm\'s length from the mic.',
             'Now step back roughly 2 m and speak a little louder.',
             'Now speak quietly, close to the mic, as if tired.']

    if args.only:
        capture(args.only, args.speaker, args.reps, hints[0])
        return

    # wake word first and with more reps - it runs always-on, so it carries
    # the highest false-accept risk and needs the most data
    if capture(WAKE, args.speaker, args.wake_reps,
               'Say "Hey Machina" naturally. Vary your pace and distance.') == 'quit':
        sys.exit(0)

    for i, w in enumerate(COMMANDS):
        if capture(w, args.speaker, args.reps, hints[i % len(hints)]) == 'quit':
            sys.exit(0)

    print('\n=== UNKNOWN CLASS (teaches the model to refuse) ===')
    d = outdir('_unknown')
    idx = next_index(d, args.speaker)
    for w in UNKNOWN:
        for k in range(args.unknown_reps):
            cmd = input(f'  say "{w}" [{k+1}/{args.unknown_reps}] ENTER (s=skip) > ').strip().lower()
            if cmd == 's':
                break
            if cmd == 'q':
                sys.exit(0)
            print('  recording...', end='', flush=True)
            x = record(CLIP_SEC)
            ok, msg = peak_ok(x)
            print(f' {msg}')
            if not ok:
                continue
            sf.write(os.path.join(d, f'{args.speaker}_{idx:03d}.wav'), x, SR)
            idx += 1

    print('\nNow record ambient noise (needed for the false-accept metric):')
    print(f'  python tools/record_dataset.py --speaker {args.speaker} --noise')


if __name__ == '__main__':
    main()
