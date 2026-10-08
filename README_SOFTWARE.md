# Software pipeline — run order

All steps run on a laptop. No hardware required.

## 0. Environment (once)

```powershell
cd "D:\AI Speech-Guided Edge Voice Assistant for Machine Operation"
.\.venv\Scripts\Activate.ps1
pip install tensorflow librosa soundfile sounddevice matplotlib tqdm
```

## 1. Verify the safety logic (works right now, no data needed)

```powershell
python tools\test_fsm.py
```
Expect 31/31 passing. Screenshot this for the review.

## 2. Record the dataset

```powershell
python tools\record_dataset.py --speaker archit
python tools\record_dataset.py --speaker archit --noise
```
Repeat for every speaker. Target 8–10 speakers.
Record ambient noise several times — the false-accept metric depends on it.

## 3. Augment

```powershell
python tools\augment.py --per-clip 8
```

## 4. Train

```powershell
python tools\train_model.py
```
Outputs: `models/kws_model.keras`, `confusion_matrix.png`, `training_curve.png`,
`report.json` with held-out accuracy.

## 5. Live demo

```powershell
python demo\live_demo.py
```
Keys: `E` e-stop, `G` guard door, `R` reset, `Q` quit.

## 6. Evaluation numbers for the report

```powershell
python tools\evaluate.py
```
Outputs: `det_curve.png`, `threshold_table.md`, `evaluation.json`.

## File map

| File | Purpose |
|---|---|
| `src/features.py` | MFCC front-end — shared by training and inference |
| `src/intent_fsm.py` | Safety state machine, seven defence layers |
| `tools/test_fsm.py` | 31 safety tests, no hardware |
| `tools/record_dataset.py` | Dataset collection |
| `tools/augment.py` | Noise mixing and perturbation |
| `tools/train_model.py` | Speaker-disjoint training + confusion matrix |
| `tools/evaluate.py` | DET curve, FAR/hour, latency |
| `demo/live_demo.py` | Full system on a laptop |
