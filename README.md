# EVSOD: DREAM Final Submission Reproduction

This repository is the standalone release for **DREAM** (**D**ensity-**R**outed
**E**vent **A**ttention **M**emory), the final EV-UAV Challenge 2 submission.
It contains the inference/training source code,
dataset loaders, frozen checkpoints, M124 verifier artifact, raw-score audit
materials and submission validators needed for an independent review.

`M244` is retained below only as the internal experiment and official archive
identifier, so that the released package remains traceable to the platform
submission and its SHA-256 fingerprint.

The official hidden-test result retained for M244 was:

| Score | Pd | Fa | IoU | Acc |
| ---: | ---: | ---: | ---: | ---: |
| **0.9347** | 0.9318 | 5.56e-06 | 0.9065 | 0.9694 |

The official test labels are not public and are not contained in this
repository. Therefore the platform score cannot be recomputed locally; the
event-level submission package, input ordering and fixed no-label policy can
be reproduced and audited exactly.

## Fast reproduction first

This is the recommended path for competition review. It requires the released
test NPZ files and does **not** require a GPU or any training. It reconstructs
M244 from the versioned M233 package and the frozen M243 audit report, then
checks all five output columns against every input event.

```bash
git clone https://github.com/Picasso9jiu/CSIG.git EVSOD
cd EVSOD
git lfs pull

conda activate EV39
export DATA_ROOT="/path/to/测试集"  # this directory contains test/test_000.npz ... test/test_030.npz
bash scripts/run_m244_fast.sh
```

Expected output:

```text
valid_txt_files: 31
total_events: 2265422
total_positive_events: 87706
sha256: 390fa26a200bb80f4729318011484621eb998269db616b313b7f022620f60e20
```

The produced archive is `outputs/m244_empty_top4_raw_test.zip`. To validate an
existing archive independently:

```bash
python scripts/verify_submission.py \
  --zip artifacts/m244_reference_submission.zip \
  --test-root "$DATA_ROOT/test" \
  --m244
```

The fast route uses these versioned release materials:

```text
artifacts/m233_base_submission.zip        # frozen M233 prediction package
artifacts/m243_raw_scores/*.npy           # frozen no-label raw scores
artifacts/m243_test_actual.json           # fixed Top-4 audit report
artifacts/m244_reference_submission.zip   # canonical official archive
```

`scripts/audit_m243_raw_scores.py` can regenerate the report from the raw
scores and input events. The expected selected indices are checked by the
script and are not entered manually.

## Method overview

DREAM is an event-level small-object detector based on a density-routed,
bidirectional temporal-memory network. M10 is the low-event expert; M26 is
the high-event temporal-memory model with temporal self-attention and a
bounded target-flow alignment head. The high-event branch uses a `0.75/0.25`
original/half-bin phase blend with the fixed M111 phase specialist.

The score stream passes P6 density/polarity thresholds, P0/P0c component
filtering, P18 weak-track recovery and the P32 track-quality bonus. Finally,
the frozen M124 background-component verifier removes persistent background.
Its release thresholds are chosen only from full-video observable statistics:

| Domain | Rule from input events | M124 threshold |
| --- | --- | ---: |
| low/middle | `event_count <= 200000` | `0.94` |
| H1 | `event_count > 200000` and polarity minority ratio `< 0.20` | `0.62` |
| other H2 | remaining high-density videos | `0.65` |
| extreme H2 | other H2 with `event_count >= 500000` | `0.63` |

M244 makes one final conservative no-label recovery: in each extreme-H2
video, recover the top raw-score event from the four selected empty
50-unit temporal bins. On the released test set this changes only eight
labels in `test_022` and `test_023`. It does not inspect target IDs, video
names, validation labels or hidden test labels.

More detail and the exact release provenance are in
[docs/M244_RELEASE_NOTES.md](docs/M244_RELEASE_NOTES.md). The complete
experiment history is retained in [note.md](note.md); it includes accepted
and rejected directions so later work does not repeat failed probes.

## Environment

The release was verified in WSL/Ubuntu with Python 3.9, PyTorch 1.9.1 + CUDA
11.1, torchvision 0.10.1, `spconv-cu111`, NumPy 1.23.5 and a CUDA 11.x
Toolkit. GPU inference requires a successful HAIS_OP build.

```bash
conda create -n EV39 python=3.9 pip -y
conda activate EV39

python -m pip install --upgrade pip
python -m pip install \
  torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirements.txt

sudo apt update
sudo apt install -y build-essential libsparsehash-dev ninja-build
cd lib/hais_ops
python setup.py build_ext develop
cd ../..
```

For a new shell, activate the environment and export the runtime paths:

```bash
conda activate EV39
export PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
python -c "import torch, HAIS_OP; import spconv.pytorch; print(torch.cuda.is_available(), 'HAIS_OP: ok')"
```

Large binary assets use Git LFS. Install it before cloning if necessary:

```bash
git lfs install
git lfs pull
```

## Dataset layout and processing

The EV-UAV data are not redistributed. Download them from the official source
and retain the original NPZ layout. Dataset parsing, sparse voxelization and
temporal sampling code are under `dataset/`; no precomputed frames are needed.

```text
训练集、验证集/
|-- train/                         # official training NPZ files
|-- val/                           # official validation NPZ files
`-- val_Challenge2.py

测试集/
`-- test/                          # test_000.npz ... test_030.npz
```

The full-checkpoint script consumes `DATA_ROOT/test/`. The retraining script
consumes `DATA_ROOT/train/` and `DATA_ROOT/val/`. The submitted TXT format is:

```text
x y t p label
```

`scripts/verify_submission.py` verifies row count, field count, original
`x/y/t/p` order and binary labels for every file before a package is accepted.

## Full inference from checkpoints

This path regenerates the M233 score stream with the released M10, M26 and
M111 checkpoints, applies the fixed no-label routing policy, recomputes the
M243 raw-score audit and then builds M244. It needs a CUDA environment and is
slower than the fast reproduction route.

```bash
conda activate EV39
export DATA_ROOT="/path/to/测试集"
bash scripts/run_m244_from_checkpoints.sh
```

Outputs are written below `outputs/m244_from_checkpoints/`. The script does
not reuse the reference submission. It uses `scripts/split_m244_domains.py`
to make the route explicit and writes a manifest containing each observed
event count, polarity ratio and selected threshold. Minor low-level numeric
differences can occur across CUDA/spconv builds; the packaged fast route is
the canonical byte-for-byte release check.

## Checkpoints and M111 averaging

All required release artifacts and their SHA-256 values are listed in
[docs/CHECKSUMS.sha256](docs/CHECKSUMS.sha256). The inference path requires
M10, M26, M111 and the M124 verifier. The M4/M13/M15/M20 checkpoints support
the documented M26 retraining chain. The three M111 seed checkpoints are
also included so the released average can be semantically rechecked:

```bash
python scripts/build_m111_average.py \
  --checkpoint checkpoints/m111_phase_seed72_epoch_001.pt \
  --checkpoint checkpoints/m111_phase_seed73_epoch_001.pt \
  --checkpoint checkpoints/m111_phase_seed76_epoch_001.pt \
  --output /tmp/m111_average.pt \
  --reference checkpoints/m111_phase_specialist_seed72_73_76_average.pt
```

The command prints `reference_model_state: ok`. The averaged tensors are
computed in `float64` and cast back to their original dtype; serialisation
metadata may make the newly written file's byte hash differ from the bundled
reference checkpoint even when every model tensor is identical.

## Full training chain

The versioned main-model chain is:

```text
M4+DACC+M5 -> M13 epoch 003 -> M15 epoch 008 -> M20 epoch 003 -> M26 epoch 003
```

All commands, epoch selection and fixed hyperparameters are in
`scripts/retrain_m26_chain.sh`. It begins from the included M4 checkpoint and
trains on `train/` only; it is deliberately long-running.

```bash
conda activate EV39
export DATA_ROOT="/path/to/训练集、验证集"
export RUN_ROOT="$PROJECT_DIR/outputs/retrain_m26"
bash scripts/retrain_m26_chain.sh
```

Use the epoch-003 M26 output in the full-inference script only after its
checkpoint has been independently evaluated. M10 and the three one-epoch
M111 specialists are fixed release experts; their frozen and averaged weights
are versioned here to make the final M244 pipeline directly reproducible.

## Repository layout

```text
checkpoints/       M4/M10/M13/M15/M20/M26/M111 release weights
artifacts/         M124 verifier, M233 base, M243 audit inputs, canonical M244 ZIP
configs/           fixed YAML and command-line override loader
dataset/           official NPZ parsing and temporal sampling code
lib/hais_ops/      CUDA extension source (build locally)
model/             EV-SpSegNet and temporal-memory architectures
utils/             inference, post-processing, M124 and evaluation utilities
scripts/           fast rebuild, full inference, validation and training entry points
note.md             chronological experiment log
```

## Attribution

This release builds on the ICCV 2025 EV-UAV official baseline. See
[NOTICE.md](NOTICE.md) for citation and dataset-attribution details.
