"""
train_model.py - train the keyword-spotting model on the collected dataset.

Run:
    python tools/train_model.py
    python tools/train_model.py --test-speakers 1
    python tools/train_model.py --held nalin          # choose the held-out speaker

Reads  data/augmented/<label>/*.wav   (falls back to data/raw if absent)
Writes models/kws_model.keras, models/labels.json, models/report.json,
       models/confusion_matrix.png, models/training_curve.png

Splits
  test        whole speakers held out (never seen in training). The silence class
              has no speaker, so 15% of silence clips are held out at random.
  validation  carved from the training speakers, grouped by source recording so an
              augmented copy of a clip can never sit in both train and validation.
"""

import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import load_mfcc, INPUT_SHAPE  # noqa: E402

DATA_AUG = os.path.join('data', 'augmented')
DATA_RAW = os.path.join('data', 'raw')
OUT = 'models'
SILENCE_SPEAKER = 'sil'          # augment.py names generated silence clips sil_0001.wav


def speaker_of(path):
    """<speaker>_<nnn>[_augN|_clean].wav -> speaker is the first field."""
    return os.path.basename(path).split('_')[0]


def base_id(path):
    """Source recording id: strips the _augN / _clean suffix added by augment.py."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'_(aug\d+|clean)$', '', stem)


def collect(root):
    labels = sorted(d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d)) and d != '_noise')
    files, ys, spk = [], [], []
    for li, lab in enumerate(labels):
        for f in glob.glob(os.path.join(root, lab, '*.wav')):
            files.append(f)
            ys.append(li)
            spk.append(speaker_of(f))
    return labels, np.array(files), np.array(ys), np.array(spk)


def make_split(spk, files, test_speakers=1, forced=None, seed=42):
    """Return (test_mask, held_out_speakers, mode).

    mode is 'speaker-disjoint' (>=2 real speakers) or 'recording-disjoint' (one speaker:
    whole source recordings are held out, so no augmented copy leaks into training)."""
    rng = np.random.RandomState(seed)
    spk = np.asarray(spk)
    real = sorted(set(str(s) for s in spk) - {SILENCE_SPEAKER})
    sil_idx = np.where(spk == SILENCE_SPEAKER)[0]

    held = None
    if forced:
        missing = [s for s in forced if s not in real]
        if missing:
            sys.exit(f'--held {missing} not found. Speakers in the dataset: {real}')
        held = set(forced)
    elif len(real) > test_speakers:
        held = set(str(s) for s in rng.choice(real, test_speakers, replace=False))

    if held is not None:
        test = np.array([str(s) in held for s in spk])
        if len(sil_idx):
            n = max(1, int(0.15 * len(sil_idx)))
            test[rng.choice(sil_idx, n, replace=False)] = True
        return test, sorted(held), 'speaker-disjoint'

    groups = np.array([base_id(f) for f in files])
    uniq = np.unique(groups)
    held_groups = rng.choice(uniq, max(1, len(uniq) // 5), replace=False)
    return np.isin(groups, held_groups), [], 'recording-disjoint'


def make_val_split(train_files, seed=42, frac=0.15):
    """Validation mask over the training files, grouped by source recording."""
    rng = np.random.RandomState(seed)
    groups = np.array([base_id(f) for f in train_files])
    uniq = np.unique(groups)
    n = max(1, int(frac * len(uniq)))
    val_groups = rng.choice(uniq, n, replace=False)
    return np.isin(groups, val_groups)


def build_model(n_classes):
    from tensorflow import keras
    from tensorflow.keras import layers as L

    def ds_block(x, filters, stride=1):
        x = L.SeparableConv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
        x = L.BatchNormalization()(x)
        return L.ReLU()(x)

    inp = keras.Input(shape=INPUT_SHAPE)
    x = L.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    x = ds_block(x, 48)
    x = ds_block(x, 64, stride=2)
    x = ds_block(x, 64)
    x = ds_block(x, 96, stride=2)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inp, out, name='kws_dscnn')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--test-speakers', type=int, default=1)
    ap.add_argument('--held', nargs='+', help='speaker name(s) to hold out, e.g. --held nalin')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    root = DATA_AUG if os.path.isdir(DATA_AUG) and os.listdir(DATA_AUG) else DATA_RAW
    if not os.path.isdir(root):
        sys.exit('No dataset found. Run tools/record_dataset.py first.')
    print(f'dataset: {root}')

    labels, files, y, spk = collect(root)
    if not len(files):
        sys.exit('No wav files found.')
    real_speakers = sorted(set(str(s) for s in spk) - {SILENCE_SPEAKER})
    print(f'{len(files)} clips | {len(labels)} classes | real speakers: {real_speakers}')
    for li, lab in enumerate(labels):
        print(f'   {lab:<14} {int((y == li).sum())}')

    # ---------------- split
    test_mask, held, mode = make_split(spk, files, args.test_speakers, args.held, args.seed)
    if mode == 'speaker-disjoint':
        print(f'\nSPLIT: speaker-disjoint. Held-out test speaker(s): {held}')
    else:
        print('\nSPLIT: recording-disjoint. Only one real speaker, so the test clips are unseen '
              'RECORDINGS of the SAME voice. This is NOT accuracy on a new speaker.')

    # ---------------- features
    print('\nextracting MFCC features...')
    X = np.stack([load_mfcc(f) for f in files])[..., None]
    print(f'feature tensor: {X.shape}')

    train_files = files[~test_mask]
    Xtr, ytr = X[~test_mask], y[~test_mask]
    Xte, yte = X[test_mask], y[test_mask]
    val_mask = make_val_split(train_files, args.seed)
    Xval, yval = Xtr[val_mask], ytr[val_mask]
    Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
    print(f'train {len(Xfit)} | val {len(Xval)} | test {len(Xte)}')
    if not len(Xte):
        sys.exit('Test set is empty.')

    # ---------------- train
    from tensorflow import keras
    model = build_model(len(labels))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    os.makedirs(OUT, exist_ok=True)
    cbs = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
    ]
    hist = model.fit(Xfit, yfit, validation_data=(Xval, yval),
                     epochs=args.epochs, batch_size=args.batch, callbacks=cbs, verbose=2)

    # ---------------- evaluate
    probs = model.predict(Xte, verbose=0)
    pred = probs.argmax(1)
    acc = float((pred == yte).mean())
    tag = 'HELD-OUT-SPEAKER' if mode == 'speaker-disjoint' else 'UNSEEN-RECORDINGS, SAME SPEAKER'
    print(f'\n{tag} TEST ACCURACY: {acc * 100:.2f}%')

    cm = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1

    per_class = {}
    print('\nper-class recall')
    for i, lab in enumerate(labels):
        tot = int(cm[i].sum())
        rec = round(float(cm[i, i] / tot), 4) if tot else None
        per_class[lab] = {'support': tot, 'recall': rec}
        print(f'   {lab:<14} {tot:>4} clips   recall {rec if rec is None else f"{rec * 100:.0f}%"}')

    model.save(os.path.join(OUT, 'kws_model.keras'))
    json.dump(labels, open(os.path.join(OUT, 'labels.json'), 'w'), indent=2)
    json.dump({'test_accuracy': acc, 'split_mode': mode, 'held_out': held,
               'train_speakers': [s for s in real_speakers if s not in held],
               'n_clips': int(len(files)), 'n_test_clips': int(len(Xte)),
               'speakers': real_speakers, 'per_class': per_class,
               'confusion_matrix': cm.tolist(), 'labels': labels},
              open(os.path.join(OUT, 'report.json'), 'w'), indent=2)

    # ---------------- plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha='center', va='center',
                            color='white' if norm[i, j] > 0.5 else 'black', fontsize=8)
        ax.set_xlabel('predicted')
        ax.set_ylabel('actual')
        who = ', '.join(held) if held else 'unseen recordings, same speaker'
        ax.set_title(f'Confusion matrix - held out: {who} ({acc * 100:.1f}%)')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'confusion_matrix.png'), dpi=140)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist.history['accuracy'], label='train')
        ax.plot(hist.history['val_accuracy'], label='validation')
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax.legend()
        ax.grid(alpha=.3)
        ax.set_title('Training curve')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'training_curve.png'), dpi=140)
        print('saved confusion_matrix.png and training_curve.png')
    except Exception as e:
        print('plotting skipped:', e)

    print(f'\nmodel -> {OUT}/kws_model.keras')
    print('next:  python tools/evaluate.py   then   python demo/live_demo.py')


if __name__ == '__main__':
    main()