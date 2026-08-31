# DREAM Release Notes (M244 Archive)

**DREAM** means **D**ensity-**R**outed **E**vent **A**ttention **M**emory.
`M244` remains the immutable internal/official archive identifier used in
filenames and SHA-256 checks.

## Final result

The official platform retained M244 as this release's final submission. The
hidden-label result was `Score=0.9347`, `Pd=0.9318`, `Fa=5.56e-06`,
`IoU=0.9065`, and `Acc=0.9694`. Hidden test labels are not included in this
repository, so the scalar score itself can only be checked by the platform.

The canonical submission archive is
`artifacts/m244_reference_submission.zip`:

```text
SHA-256: 390fa26a200bb80f4729318011484621eb998269db616b313b7f022620f60e20
31 TXT files, 2,265,422 events, 87,706 positive events
```

## Inference chain

M244 is a conservative extension of M233. The upstream predictor contains:

1. M10 for videos with at most 30,000 events and M26 for the remaining
   videos.
2. M26 is a bidirectional temporal-memory network with temporal attention and
   a bounded target-flow alignment head.
3. M111 is a float64 parameter average of three one-epoch phase specialists;
   it supplies the shifted half-bin branch, blended with the M26 original
   stream at `0.25/0.75`.
4. P6 density/polarity decision thresholds, P0/P0c component filtering,
   P18 one-event weak-track recovery, and P32 track-quality score bonus.
5. Frozen M124 component-background verifier. Its threshold is selected from
   observable full-video statistics only:

| Domain | Observable rule | M124 threshold |
| --- | --- | ---: |
| Low/middle | event count `<= 200000` | `0.94` |
| H1 | event count `> 200000` and polarity minority ratio `< 0.20` | `0.62` |
| Other H2 | remaining high-density domain | `0.65` |
| Extreme H2 | other H2 with event count `>= 500000` | `0.63` |

M244 then restores one raw-score maximum from each of the four highest ranked
otherwise empty 50-unit temporal bins in each extreme-H2 video. This is a
fixed, label-free rule. In the released test set it restores four events in
each of `test_022` and `test_023`, eight events total; all other prediction
bytes remain M233-identical.

The supporting M243 audit material is versioned under `artifacts/` so this
last transformation is inspectable rather than a hand-edited submission:

```text
m233_base_submission.zip
m243_raw_scores/test_022.npy
m243_raw_scores/test_023.npy
m243_test_actual.json
m244_reference_submission.zip
```

`scripts/audit_m243_raw_scores.py` recomputes the Top-4 indices from frozen
scores. `scripts/rebuild_m244.py` applies them to M233 and validates every
row against the public NPZ event order.
