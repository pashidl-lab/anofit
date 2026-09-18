# Third-party notices

anofit includes source code from the projects below. Each keeps its own license,
and those licenses grant you rights that anofit's own LICENSE does not restrict.

Apache License 2.0 requires that modified files carry a notice saying they changed.
**Section "What we changed" under each project is that notice.** The measured numbers
cited there are from our own runs on MVTec LOCO; they are given so a reader can tell a
deliberate change from an accident.

License texts are in `licenses/`:

| File | Applies to |
|---|---|
| `licenses/Apache-2.0.txt` | GroundingDINO · Segment Anything · SAM-HQ · Recognize Anything · CSAD |
| `licenses/MIT-SALAD.txt` | SALAD |

---

## 1. GroundingDINO

- Source: <https://github.com/IDEA-Research/GroundingDINO>
- Copyright (c) 2023 IDEA. All Rights Reserved.
- License: Apache License 2.0
- Location in this package: `anofit/train/logic/groundingdino/`
- Vendored: 32 Python files

### What we changed

**1 file changed, 31 unchanged.**

`groundingdino/models/GroundingDINO/bertwarper.py` — made it work on transformers 5.x.

- transformers 5.x removed the head-mask helpers from `ModuleUtilsMixin`, and this file
  borrows `get_head_mask` from HuggingFace's `BertModel`. Without them it fails with
  `AttributeError: 'BertModel' object has no attribute 'get_head_mask'`. We restore the
  missing names through `anofit/train/logic/_transformers_compat.py`.
- `get_extended_attention_mask` lost its `device` parameter in 5.x, so the third
  positional argument landed in the `dtype` slot and died with
  `TypeError: to() received an invalid combination of arguments`. We drop the argument,
  which behaves identically on 4.x where it was already unused.

Both edits are marked in the file between `── anofit 수정 ──` comments.

---

## 2. Segment Anything (SAM) and SAM-HQ

- Source: <https://github.com/facebookresearch/segment-anything> and
  <https://github.com/SysCV/sam-hq>
- Copyright (c) Meta Platforms, Inc. and affiliates.
- License: Apache License 2.0
- Location in this package: `anofit/train/logic/segment_anything/`
- Vendored: 17 Python files

### What we changed

**Nothing. All 17 files are byte-identical to the version we vendored.**

Note that `SamAutomaticMaskGenerator` is constructed with fixed arguments at the call
site in `anofit/train/logic/pseudo_label.py` (`points_per_side=32`,
`points_per_batch=16`, `crop_n_layers=1`, …). That is our code, not a change to SAM.

---

## 3. Recognize Anything (RAM++ / Tag2Text)

- Source: <https://github.com/xinyu1205/recognize-anything>
- Written by Xinyu Huang and contributors.
- License: Apache License 2.0
- Location in this package: `anofit/train/logic/_ram/`

CSAD imported this as an installed package (`from ram.models import ram_plus`). We
vendored it under `_ram` so that a user does not have to install a separate package
whose import name collides with other things.

### What we changed

**Vendored as-is; the package was renamed `ram` → `_ram` and imports adjusted to match.**
No change to the models or to the tagging logic.

---

## 4. CSAD — Unsupervised Component Segmentation for Logical Anomaly Detection

- Source: <https://github.com/Tokichan/CSAD> (BMVC 2024, arXiv:2408.15628)
- License: Apache License 2.0
- Location in this package: `anofit/train/logic/`

This is the logical branch: pseudo-labels from GroundingDINO + SAM-HQ, component
clustering, a segmentor, and patch histograms.

### What we changed

| File | Changed lines | What and why |
|---|---:|---|
| `pseudo_label.py` | ~580 | The bulk of our work. See below. |
| `models/segmentation/dataloader.py` | 25 | `sorted(glob(...))` on the four index-paired lists, plus a guard that warns when their lengths differ. Windows returns names in order and Linux does not, so without sorting the image and its label drift apart — measured −8.1 AUROC. |
| `ram_prompt.py` | 29 | Import from the vendored `_ram`; write the prompt where the rest of the pipeline reads it. |
| `models/segmentation/segmentor.py` | 10 | Removed the AMP path (it was off by default, `SEG_AMP=0`); behaviour unchanged. |
| `models/segmentation/lsa.py` | 10 | Return early when a frame has no component masks, instead of indexing an empty list. |
| `CSAD.py`, `export_model.py` | 2 each | Import paths. |

`train_light.py`, `train_full.py`, `preview_segmentation.py`, `extract_prompt.py`,
`_transformers_compat.py` and everything under `anofit/train/` outside `logic/` are ours.

#### `pseudo_label.py` in detail

- **Deterministic random projection.** The 2048→512 projection used
  `normal_(0, 0.01)` with no seed at all, and that draw decides the MeanShift cluster
  count — we measured it swing between 7 and 4 across runs. We draw it from a dedicated
  `torch.Generator` so it does not depend on how much RNG the preceding RAM++/GDINO/SAM
  calls happened to consume.
- **Background decision.** The original matched a background phrase as a substring with
  no `strip()`, so `' container .'` never matched `container tray(0.45)` and the setting
  silently did nothing for all 372 images of one category. We strip the tokens, add a
  geometric fallback, and split the containment test into "how much of this mask is
  inside the region" and "how much of the region does it cover" — in a containment the
  original symmetric ratio is 1.0 for both the part and the background, so direction
  alone cannot tell them apart.
- **Cache validity.** The generation step was skipped whenever the output folder
  existed, so an interrupted run left partial masks that were then trained on, and
  changing a threshold had no effect. We record a fingerprint of the generating settings
  and check per-image completeness.
- **Crash guards.** Removing every detection as background left `torch.stack([])`;
  a missing background file made every mask count as background and deleted them all.
- **Colour palette** extended from 20 to 256 entries (visualisation only) because
  precise segmentation produces more than 20 clusters and the run died at the last step.

---

## 5. SALAD

- Source: <https://github.com/matic-fucka/SALAD> (ICCV 2025, arXiv:2509.02101)
- Copyright (c) 2025 Matic Fučka
- License: MIT
- Location in this package: `anofit/train/struct/`

This is the structural branch: foreground masks, composition maps, and the
teacher/student/autoencoder training.

### What we changed

| File | Changed lines | What and why |
|---|---:|---|
| `create_pseudo_labels.py` | 93 | Skip frames whose mask holds no object pixels, and resample when a frame has fewer pixels than the sample size — both used to end as an opaque numpy `ValueError` several frames deep. Cast features to float64 explicitly so sklearn's k-means does not fail one stage later. |
| `train_salad.py` | 81 | Average AUROC over the defect classes that are actually present; the original assumed both logical and structural exist and produced NaN when only one did, which erased a correctly computed score. Emit progress markers. Tolerate a test set with no defects. |
| `salad_dataset.py` | 16 | numpy 2 compatibility for imgaug (`np.sctypes` was removed), and guards for frames with no components. |
| `create_fg_masks.py` | 7 | Take the model directory from an environment variable; sort the glob. |

`train_entry.py` (the four-stage orchestrator), `salad_infer.py` and `_numpy_compat.py`
are ours. Nine other files are unchanged.

---

## 6. Model weights are not included

anofit does not redistribute any model weights. `anofit fetch` downloads them from their
original publishers and verifies a SHA-256 for each:

| Weight | Publisher | License (as declared by the publisher, checked 2026-09-14) |
|---|---|---|
| `groundingdino_swint_ogc.pth` | IDEA-Research (GitHub release) | Apache 2.0 |
| `sam_vit_h_4b8939.pth` | Meta (dl.fbaipublicfiles.com) | Apache 2.0 |
| `sam_hq_vit_h.pth` | SysCV (Hugging Face `lkeab/hq-sam`) | Apache 2.0 |
| `ram_plus_swin_large_14m.pth` | Xinyu Huang et al. (Hugging Face `xinyu1205/recognize-anything-plus-model`) | Apache 2.0 |
| `teacher_medium.pth` | Matic Fučka, SALAD repository (`models/teacher_medium.pth`); the teacher was distilled by nelson1425/EfficientAD | MIT (SALAD) / Apache 2.0 (EfficientAD) |
| `imagenette2.tgz` | fast.ai (Imagenette) — used as the ImageNet stand-in for the EfficientAD penalty | Apache 2.0 for the dataset packaging; the images are ImageNet images and remain under the ImageNet terms of access |

The exact URLs and SHA-256 values are in `anofit/fetch_manifest.json`. Each weight carries its
own license, which you accept with its publisher, not with us.

Also downloaded on first training run, by the libraries themselves into their own caches
(`HF_HOME`, `TORCH_HOME`): `bert-base-uncased` (GroundingDINO's text encoder),
`timm/wide_resnet50_2` (CSAD encoder), `dino_vitbase8` (SALAD), and — since 0.1.1 —
`timm/vit_small_patch14_dinov2.lvd142m` (DINOv2 ViT-S/14, Meta AI, Apache 2.0, ~85 MB;
used by the `dpat` and `dhist` branches). These are not part of `anofit fetch`; a first training needs
network access for them (about 1.1 GB). anofit copies the DINOv2 weights into
`weights_root/timm__.../encoder.pth` on first use so an offline site can be served by
copying that folder.

---

## 7. Runtime dependencies

Installed separately by pip and not redistributed here: NumPy (BSD-3-Clause),
OpenCV (Apache 2.0), Pillow (MIT-CMU), PyYAML (MIT), PyTorch (BSD-3-Clause),
torchvision (BSD-3-Clause), and — for training — timm (Apache 2.0),
transformers (Apache 2.0), scikit-learn (BSD-3-Clause), SciPy (BSD-3-Clause).
