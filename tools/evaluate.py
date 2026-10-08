"""
evaluate.py — the numbers that go in the report.

Run:
    python tools/evaluate.py

Produces:
    models/det_curve.png        false rejects vs false accepts per hour
    models/threshold_table.md   operating points, for the slide
    models/evaluation.json      machine-readable results

Why a DET curve and not just accuracy: in keyword spotting the question is
not "how often is it right" but "how often does it act when it should not".
Accuracy hides that. A DET curve shows the whole trade-off, so a threshold
can be chosen deliberately rather than by default.
"""

import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import load_mfcc, mfcc, SR  # noqa: E402

OUT = 'models'
WINDOW_S = 1.0
STRIDE_S = 0.25


def speaker_of(p):
    return os.path.basename(p).split('_')[0]


def main():
    labels = json.load(open(os.path.join(OUT, 'labels.json')))
    report = json.load(open(os.path.join(OUT, 'report.json')))
    held = set(report.get('held_out', []))

    from tensorflow import keras
    model = keras.models.load_model(os.path.join(OUT, 'kws_model.keras'))

    # ---------------------------------------------------------- positives
    root = 'data/raw' if os.path.isdir('data/raw') else 'data/augmented'
    pos_files, pos_y = [], []
    for li, lab in enumerate(labels):
        if lab.startswith('_'):
            continue
        for f in glob.glob(os.path.join(root, lab, '*.wav')):
            if not held or speaker_of(f) in held:
                pos_files.append(f)
                pos_y.append(li)
    if not pos_files:
        sys.exit('No positive clips found. Record data first.')
    print(f'{len(pos_files)} command clips for FRR measurement')

    X = np.stack([load_mfcc(f) for f in pos_files])[..., None]
    t0 = time.perf_counter()
    P = model.predict(X, verbose=0)
    per_clip_ms = (time.perf_counter() - t0) / len(X) * 1000
    pos_conf = P.max(1)
    pos_pred = P.argmax(1)
    correct = pos_pred == np.array(pos_y)

    # ---------------------------------------------------------- negatives
    noise_files = glob.glob(os.path.join('data/raw/_noise', '*.wav'))
    neg_conf = []
    total_noise_s = 0.0
    if noise_files:
        import librosa
        for nf in noise_files:
            y, _ = librosa.load(nf, sr=SR, mono=True)
            total_noise_s += len(y) / SR
            step = int(STRIDE_S * SR)
            win = int(WINDOW_S * SR)
            chunks = [y[i:i + win] for i in range(0, max(1, len(y) - win), step)]
            if not chunks:
                continue
            F = np.stack([mfcc(c) for c in chunks])[..., None]
            Q = model.predict(F, verbose=0)
            for row in Q:
                k = int(row.argmax())
                if not labels[k].startswith('_'):       # a command was predicted
                    neg_conf.append(float(row[k]))
                    
        print(f'{total_noise_s/60:.1f} min of ambient audio scanned '
              f'({len(neg_conf)} command-like windows)')
    else:
        print('WARNING: no data/raw/_noise/*.wav - false-accept rate cannot be measured.')

    neg_conf = np.array(neg_conf) if neg_conf else np.array([])
    hours = max(total_noise_s / 3600.0, 1e-9)

    # ---------------------------------------------------------- sweep
    rows = []
    for thr in np.arange(0.50, 1.00, 0.025):
        frr = float((~(correct & (pos_conf >= thr))).mean())
        fa = int((neg_conf >= thr).sum()) if neg_conf.size else 0
        rows.append({'threshold': round(float(thr), 3),
                     'frr': round(frr, 4),
                     'false_accepts': fa,
                     'fa_per_hour': round(fa / hours, 3) if neg_conf.size else None})

    # operating point: lowest FRR that still keeps FA/hr under 0.5
    chosen = None
    for r in rows:
        if r['fa_per_hour'] is not None and r['fa_per_hour'] < 0.5:
            chosen = r
            break
    if chosen is None:
        chosen = min(rows, key=lambda r: r['frr'])

    acc = float(correct.mean())
    results = {
        'top1_accuracy': round(acc, 4),
        'n_command_clips': len(pos_files),
        'ambient_minutes_scanned': round(total_noise_s / 60, 2),
        'inference_ms_per_clip': round(per_clip_ms, 2),
        'recommended_threshold': chosen,
        'sweep': rows,
    }
    json.dump(results, open(os.path.join(OUT, 'evaluation.json'), 'w'), indent=2)

    # ---------------------------------------------------------- markdown
    md = ['| Threshold | False reject rate | False accepts/hour |',
          '|---|---|---|']
    for r in rows[::4]:
        fa = '—' if r['fa_per_hour'] is None else f"{r['fa_per_hour']:.2f}"
        md.append(f"| {r['threshold']:.2f} | {r['frr']*100:.1f}% | {fa} |")
    md.append('')
    md.append(f"**Chosen operating point: threshold {chosen['threshold']:.2f} "
              f"→ FRR {chosen['frr']*100:.1f}%, "
              f"FA/hr {chosen['fa_per_hour'] if chosen['fa_per_hour'] is not None else 'n/a'}**")
    open(os.path.join(OUT, 'threshold_table.md'), 'w').write('\n'.join(md))

    # ---------------------------------------------------------- plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        if neg_conf.size:
            ax.plot([r['fa_per_hour'] for r in rows],
                    [r['frr'] * 100 for r in rows], marker='o', ms=3, color='#e08900')
            ax.set_xlabel('false accepts per hour')
            ax.axvline(0.5, ls='--', color='#888', lw=1)
            ax.text(0.52, ax.get_ylim()[1] * 0.92, 'target < 0.5/hr', fontsize=9, color='#555')
        else:
            ax.plot([r['threshold'] for r in rows], [r['frr'] * 100 for r in rows],
                    marker='o', ms=3, color='#e08900')
            ax.set_xlabel('confidence threshold')
        ax.set_ylabel('false reject rate (%)')
        ax.set_title('Detection error tradeoff')
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'det_curve.png'), dpi=140)
        print('saved det_curve.png')
    except Exception as e:
        print('plot skipped:', e)

    print(f'\ntop-1 accuracy          {acc*100:.2f}%')
    print(f'inference per clip      {per_clip_ms:.1f} ms')
    print(f"recommended threshold   {chosen['threshold']:.2f} "
          f"(FRR {chosen['frr']*100:.1f}%, FA/hr {chosen['fa_per_hour']})")
    print(f'\nwrote {OUT}/evaluation.json, threshold_table.md, det_curve.png')


if __name__ == '__main__':
    main()
