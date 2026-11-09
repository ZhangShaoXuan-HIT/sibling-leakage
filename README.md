# Augmentation-induced sibling leakage in small-data surface imaging

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24082523.svg)](https://doi.org/10.5281/zenodo.24082523)

Reproduction package for the manuscript *"Augmentation-induced sibling leakage in
small-data surface imaging: measurement and a reporting protocol"*.

The repository reproduces every number and figure in the paper from two public
datasets. No private data, no pre-trained checkpoints, no hardware beyond a single
consumer GPU (all experiments also run on CPU, slowly).

## What the claim is

Papers in small-data surface and microscopy imaging typically expand a handful of
acquisitions into hundreds of images by augmentation, then split the augmented pool
at random. Every test image then has a near-copy — a *sibling* — in the training set,
and the reported metric measures recognition rather than generalisation. This package
measures that effect at a fixed annotation budget by comparing a random split (A)
with a split by acquisition (B).

## Repository layout

| Path | Purpose | Produces |
|---|---|---|
| `src/neuseg_data.py` | NEU-Seg loading, masks to boxes, deterministic variants, both protocols | — |
| `src/stm_data.py` | STM scans, tiling, both protocols | — |
| `src/models.py` | Faster R-CNN construction, training loop, mAP computation | — |
| `src/run_slr.py` | Sibling-leakage statistic from images alone (no training) | `out/slr_final.json` |
| `src/run_neuseg.py` | Detection sweep over source counts *k* | `out/results_curve_s*.json` |
| `src/run_stm.py` | Segmentation sweep over training scans | `out/results_stm_s012.json` |
| `src/fig1.py`, `fig2.py`, `fig3.py` | Manuscript figures | `figures/*.pdf`, `*.png` |
| `src/style.py` | Shared plotting style | — |
| `src/config.py` | Data and output paths (environment variables) | — |

## Data

Neither dataset is redistributed here. Download both, then point the environment
variables at them (see `data/README.md` for the expected directory tree):

- **NEU-Seg** — surface-defect images with pixel annotations, Mendeley Data,
  <https://doi.org/10.17632/z8g47vzt4p.1> (CC BY 4.0)
- **STM point defects** — 14 scans on Si(001)-H-AsH3, Ge(001)-AsH3 and TiO2(110),
  Zenodo, <https://doi.org/10.5281/zenodo.15525935> (CC BY 4.0)

```bash
export NEUSEG_ROOT=/path/to/NEU_Seg-main
export STM_ROOT=/path/to/Data/Segmentation_data
```

## Environment

```
Python >= 3.11
torch >= 2.1, torchvision >= 0.16
numpy, scipy, Pillow, matplotlib
```

`pip install -r requirements.txt`. All experiments in the paper ran on one
RTX 4060 Laptop GPU with 8 GB of memory.

## Reproducing the results

```bash
# 1. leakage statistic, no training (seconds to a minute)
python src/run_slr.py
#    -> out/slr_final.json
#    expected: k=2 0.97, k=3 0.93, k=5 0.82, k=10 0.68 for protocol A; 0.00 for B

# 2. detection sweep, one partition (~16 min per run, 8 runs)
python src/run_neuseg.py --ks 2,3,5,10 --protocols A,B --epochs 30 --seed 0 --tag curve_s0
#    repeat with --seed 1 and --seed 2 for the three partitions in the paper

# 3. segmentation sweep, three partitions (~21 s per run, 18 runs)
python src/run_stm.py --ks 2,3,5 --seeds 0,1,2 --epochs 30 --tag stm_s012

# 4. figures
python src/fig1.py && python src/fig2.py && python src/fig3.py
```

## Expected results

Detection (NEU-Seg, mean ± s.d. over three partitions, 240 training images):

| *k* | A mAP@0.5:0.95 | B mAP@0.5:0.95 | gap |
|---|---|---|---|
| 2 | 0.981 ± 0.006 | 0.435 ± 0.074 | +0.546 |
| 3 | 0.955 ± 0.013 | 0.411 ± 0.104 | +0.544 |
| 5 | 0.910 ± 0.019 | 0.455 ± 0.072 | +0.455 |
| 10 | 0.800 ± 0.047 | 0.497 ± 0.090 | +0.303 |

Segmentation (STM, Dice on 400 held-out patches):

| *k* | A | B | gap |
|---|---|---|---|
| 2 | 0.783 ± 0.032 | 0.360 ± 0.106 | +0.424 |
| 3 | 0.767 ± 0.010 | 0.320 ± 0.039 | +0.448 |
| 5 | 0.725 ± 0.050 | 0.447 ± 0.070 | +0.278 |

Small differences in the last digit are expected: GPU non-determinism and library
versions move mAP by a few thousandths. The claims are at the level of the gap, not
the fourth decimal.

## Notes on the protocols

- A *source* is one acquisition: one NEU-Seg image, or one STM scan.
- Protocol A pools the variants of all *k* sources per class (including the reserved
  test sources) and splits the pool at random.
- Protocol B reserves one source per class before sampling and trains only on the
  remaining *k* − 1; the reserved sources supply the test set.
- The annotation budget is fixed at 240 images (NEU-Seg) or 400 patches (STM) for
  every configuration, so only the provenance of the training data changes.

## Licence

Code: MIT. The two datasets keep their own licences (CC BY 4.0).
