# Lunar crater detection from DEM data

Detecting impact craters on the Moon from digital elevation models with a U-Net,
compared against a classical edge-detection baseline, and checked for the
detection bias that would distort crater-based surface dating.

Course project, Machine Learning, Institute of Geosciences, University of
Potsdam. Paul Schrepel and Khin San Wai.

---

## Task

The DEM tiles come with a catalogue of known craters. Two questions:

1. Can a U-Net find craters accurately enough to reproduce the crater
   size-frequency distribution that surface dating is based on?
2. How does it compare with a classical pipeline of Sobel edges plus template
   matching?

The network predicts crater **rims** as a binary mask. Turning that mask into a
list of craters with centre and radius is a separate step, and two extraction
methods are compared for it.

---

## The notebooks

| # | notebook | content |
|---|---|---|
| 1 | [`1_training.ipynb`](1_training.ipynb) | DeepMoon data check, preprocessing, U-Net and training with Dice loss |
| 2 | [`2_evaluation.ipynb`](2_evaluation.ipynb) | crater extraction (template matching, circle-and-arc fitting, union), matching against the catalogue, baseline comparison, dating bias |
| 3 | [`3_demo.ipynb`](3_demo.ipynb) | the trained model on a 200-tile sample, end to end |

All result cells keep their stored outputs, so every number and figure is
readable on GitHub without running anything.

**Notebook 3 runs as it is.** The sample, the catalogue excerpt and the trained
weights are in this repository. Notebooks 1 and 2 need the full DeepMoon
dataset for training, see below.

---

## Setup

```bash
git clone <repo-url>
cd FinalProject
pip install -r requirements.txt
jupyter lab
```

For GPU training install the matching PyTorch build from
[pytorch.org](https://pytorch.org) first, see the comment at the top of
`requirements.txt`.

### Data

The training, validation and test data are the pre-generated DeepMoon
supplemental materials (Silburt et al., 2019), derived from a merged
LOLA–Kaguya Terrain Camera DEM. They are not part of this repository and are
available at [doi.org/10.5281/zenodo.1133969](https://doi.org/10.5281/zenodo.1133969).

Notebooks 1 and 2 expect them unpacked into `Data/`:

```
Data/
├── train_images.hdf5    input_images and target_masks, 30,000 tiles of 256x256
├── train_craters.hdf5   one catalogue table per tile
├── dev_images.hdf5      dev_craters.hdf5
└── test_images.hdf5     test_craters.hdf5
```

### Repository layout

```
FinalProject/
├── 1_training.ipynb   2_evaluation.ipynb   3_demo.ipynb
├── crater_unet_best.pt       trained weights, epoch 43
├── demo_sample.hdf5          200 DEM tiles with true masks, for notebook 3
├── demo_craters.hdf5         the catalogue entries of those tiles
├── requirements.txt
└── README.md
```

---

## Method

**U-Net.** DEM input normalised to 0–1, 30,000 training tiles, batch size 16,
50 epochs, Adam. The loss is the Dice loss. The weights of epoch 43, the minimum of the validation Dice loss,
are the ones used everywhere else. A variant with BCE added to the Dice loss
was tried as well; the Dice-only run provided the weights used here.

**Extraction.** Method A binarises the predicted mask and slides ring templates
of every radius over it. Method B fits circles to clusters of rim pixels with
Levenberg–Marquardt, removes them, and recovers the weaker broken rims by arc
fitting. Method C is the union of both.

**Matching.** A detection counts as correct if centre and radius agree with a
catalogue entry within a tolerance that scales with the crater radius, so large
and small craters are judged on the same footing.

**Baseline.** The same extraction, but on Sobel edges of the smoothed raw DEM
instead of the U-Net mask.

---

## Results

The three extraction methods on 2,000 test tiles, matched against the catalogue:

| pixel band | method | precision | recall | F1 |
|---|---|---|---|---|
| 1–100 px | A: template matching | 0.781 | 0.212 | 0.333 |
| 1–100 px | B: circle + arc | 0.823 | 0.347 | 0.488 |
| 1–100 px | C: union | 0.764 | 0.370 | 0.499 |
| 5–50 px | A: template matching | 0.666 | 0.682 | 0.674 |
| 5–50 px | B: circle + arc | 0.795 | 0.653 | 0.717 |
| 5–50 px | C: union | 0.637 | 0.745 | 0.687 |

Against the classical baseline on 4,000 tiles, both with the union extraction:

| pixel band | method | precision | recall | F1 |
|---|---|---|---|---|
| 1–100 px | Sobel baseline | 0.099 | 0.214 | 0.135 |
| 1–100 px | U-Net | 0.764 | 0.371 | 0.500 |
| 5–50 px | Sobel baseline | 0.082 | 0.404 | 0.136 |
| 5–50 px | U-Net | 0.638 | 0.747 | 0.688 |

For dating, the relevant quantity is not F1 but how many real craters are
recovered and how many detections are produced in total:

| method | true | correct | all detections | completeness | raw ratio |
|---|---|---|---|---|---|
| baseline | 291,650 | 65,559 | 508,004 | 22.5 % | 1.74 |
| U-Net | 291,650 | 114,432 | 148,343 | 39.2 % | 0.51 |

The baseline produces 1.74 detections per true crater while finding under a
quarter of them, so a count based on it would be dominated by false positives.
The U-Net recovers almost twice as many real craters and stays below the true
count. 
Part of what counts as a false positive is a limitation of the reference, not of
the model. Notebook 2 shows tiles in which the U-Net marks craters that are
clearly visible in the DEM but absent from the catalogue.

---

## Sources

- Silburt, A., Ali-Dib, M., Zhu, C., Jackson, A., Valencia, D., Kissin, Y.,
  Tamayo, D., & Menou, K. (2019). Lunar crater identification via deep
  learning. *Icarus*, 317, 27–38. Dataset:
  [doi.org/10.5281/zenodo.1133969](https://doi.org/10.5281/zenodo.1133969)
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks
  for biomedical image segmentation. *MICCAI*.
- Python, PyTorch, NumPy, pandas, SciPy, scikit-image, scikit-learn, h5py,
  Matplotlib, Jupyter.
