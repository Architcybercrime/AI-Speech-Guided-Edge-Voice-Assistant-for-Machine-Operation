"""
train_model.py — train the keyword-spotting model on the collected dataset.

Run:
    python tools/train_model.py
    python tools/train_model.py --epochs 60 --test-speakers 2

Reads  data/augmented/<label>/*.wav   (falls back to data/raw if absent)
Writes models/kws_model.keras, models/labels.json, models/report.json,
       models/confusion_matrix.png, models/training_curve.png

The split is SPEAKER-DISJOINT: whole speakers are held out for testing, so
reported accuracy reflects unseen voices. A random split would let the model
memorise a speaker and report a number that collapses on demo day.
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import load_mfcc, INPUT_SHAPE  # noqa: E402

DATA_AUG = os.path.join('data', 'augmented')
DATA_RAW = os.path.join('data', 'raw')
OUT = 'models'


def speaker_of(path):
    """Filenames are <speaker>_<nnn>[_augN].wav -> speaker is the first field."""
    return os.path.basename(path).split('_')[0]


def collect(root):
    labels = sorted(d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d)) and d != '_noise')
    files, ys, spk = [], [], []
    for li, lab in enumerate(labels):
        for f in glob.glob(os.path.join(root, lab, '*.wav')):
            files.append(f)
            ys.append(li)
            spk.append(speaker_of(f))
    return labels, files, np.array(ys), np.array(spk)


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
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    root = DATA_AUG if os.path.isdir(DATA_AUG) and os.listdir(DATA_AUG) else DATA_RAW
    if not os.path.isdir(root):
        sys.exit('No dataset found. Run tools/record_dataset.py first.')
    print(f'dataset: {root}')

    labels, files, y, spk = collect(root)
    if not files:
        sys.exit('No wav files found.')
    speakers = sorted(set(spk))
    print(f'{len(files)} clips | {len(labels)} classes | {len(speakers)} speakers: {speakers}')
    for li, lab in enumerate(labels):
        print(f'   {lab:<14} {int((y == li).sum())}')

    # ---------------- speaker-disjoint split
    if len(speakers) > args.test_speakers + 1:
        held = set(random.sample(speakers, args.test_speakers))
        print(f'\nheld-out test speaker(s): {sorted(held)}')
        test_mask = np.array([s in held for s in spk])
    else:
        print('\nWARNING: too few speakers for a disjoint split - falling back to a '
              'random 20% split. Accuracy will be OPTIMISTIC. Record more speakers.')
        test_mask = np.zeros(len(files), bool)
        test_mask[np.random.choice(len(files), max(1, len(files) // 5), replace=False)] = True

    # ---------------- features
    print('\nextracting MFCC features...')
    X = np.stack([load_mfcc(f) for f in files])[..., None]
    print(f'feature tensor: {X.shape}')

    Xtr, ytr = X[~test_mask], y[~test_mask]
    Xte, yte = X[test_mask], y[test_mask]
    # validation carved from training speakers only
    idx = np.random.permutation(len(Xtr))
    nval = max(1, int(0.15 * len(Xtr)))
    val_i, tr_i = idx[:nval], idx[nval:]
    print(f'train {len(tr_i)} | val {len(val_i)} | test {len(Xte)}')

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
    hist = model.fit(Xtr[tr_i], ytr[tr_i], validation_data=(Xtr[val_i], ytr[val_i]),
                     epochs=args.epochs, batch_size=args.batch, callbacks=cbs, verbose=2)

    # ---------------- evaluate
    probs = model.predict(Xte, verbose=0)
    pred = probs.argmax(1)
    acc = float((pred == yte).mean())
    print(f'\nHELD-OUT TEST ACCURACY: {acc*100:.2f}%')

    cm = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1

    per_class = {}
    for i, lab in enumerate(labels):
        tot = int(cm[i].sum())
        per_class[lab] = {'support': tot,
                          'recall': round(float(cm[i, i] / tot), 4) if tot else None}

    model.save(os.path.join(OUT, 'kws_model.keras'))
    json.dump(labels, open(os.path.join(OUT, 'labels.json'), 'w'), indent=2)
    json.dump({'test_accuracy': acc, 'n_clips': len(files), 'speakers': speakers,
               'per_class': per_class, 'confusion_matrix': cm.tolist(),
               'labels': labels},
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
        ax.set_xlabel('predicted'); ax.set_ylabel('actual')
        ax.set_title(f'Confusion matrix - held-out speakers ({acc*100:.1f}%)')
        fig.tight_layout(); fig.savefig(os.path.join(OUT, 'confusion_matrix.png'), dpi=140)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist.history['accuracy'], label='train')
        ax.plot(hist.history['val_accuracy'], label='validation')
        ax.set_xlabel('epoch'); ax.set_ylabel('accuracy'); ax.legend(); ax.grid(alpha=.3)
        ax.set_title('Training curve')
        fig.tight_layout(); fig.savefig(os.path.join(OUT, 'training_curve.png'), dpi=140)
        print('saved confusion_matrix.png and training_curve.png')
    except Exception as e:
        print('plotting skipped:', e)

    print(f'\nmodel -> {OUT}/kws_model.keras')
    print('next:  python demo/live_demo.py')


if __name__ == '__main__':
    main()
