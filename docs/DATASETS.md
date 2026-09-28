# Datasets

SRRNet reads the original downloaded image and mask files. Do not copy,
rename, or generate temporal inputs before training or inference.

## Root selection

The training configs use repository-relative defaults under `data/` and accept
the following overrides:

```bash
export COD10K_ROOT=data/COD10K-v3
export MOCA_MASK_ROOT=data/MoCA_Video
export CAD_ROOT=data/CamouflagedAnimalDataset
```

`SRR_DATA_ROOT` may instead point to a common parent directory. A
dataset-specific variable takes precedence.

## COD10K-v3

Download COD10K-v3 from the
[COD10K project page](https://dengpingfan.github.io/pages/COD.html)
([paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Fan_Camouflaged_Object_Detection_CVPR_2020_paper.html),
[author repository](https://github.com/DengPingFan/SINet)) and retain this
layout:

```text
COD10K-v3/
  Train/
    Image/
      COD10K-CAM-*.jpg
    GT_Object/
      COD10K-CAM-*.png
  Test/
    Image/
      COD10K-CAM-*.jpg
    GT_Object/
      COD10K-CAM-*.png
```

`COD10KDataset` pairs images and masks by filename stem and, by default, keeps
the 3,040 camouflaged training pairs. Samples are grouped by the category in
the filename. A neighboring image and another image from the same category
provide the previous and reference training inputs.
The pre-training recipe evaluates every tenth test sample in each category.

## MoCA-Mask

Download MoCA-Mask from the
[SLT-Net project page](https://xueliancheng.github.io/SLT-Net-project/)
([MoCA-Mask paper](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Implicit_Motion_Handling_for_Video_Camouflaged_Object_Detection_CVPR_2022_paper.html),
[original MoCA paper](https://robots.ox.ac.uk/~vgg/publications/2020/Lamdouar20/),
[author repository](https://github.com/XuelianCheng/SLT-Net)). Both upstream
split names and their short aliases are accepted:

```text
MoCA_Video/
  TrainDataset_per_sq/          # Train/ is also accepted
    <sequence>/
      Imgs/<frame>.jpg
      GT/<frame>.png
  TestDataset_per_sq/           # Test/ is also accepted
    <sequence>/
      Imgs/<frame>.jpg
      GT/<frame>.png
```

`MoCAMaskDataset` matches image and mask stems, natural-sorts each sequence,
uses the immediately preceding available frame, and samples references only
from the current or earlier frames. For inference, `--data-root` may point to
`MoCA_Video/` or directly to its test split directory.

## CAD

CAD was introduced with
[It's Moving! A Probabilistic Model for Causal Motion Segmentation in Moving Camera Videos](https://arxiv.org/abs/1604.00136).
Download `CamouflagedAnimalDataset.zip` from the
[dataset link on the original author's project page](https://www.pia-bideau.com/research/structure-from-motion)
and extract it without renaming its sequence directories.
The direct layout and a common wrapper layout are both accepted:

```text
CamouflagedAnimalDataset/
  <sequence>/
    frames/<sequence>_<frame-id>.png
    groundtruth/<frame-id>_gt.png
```

```text
CamouflagedAnimalDataset/
  original_data/
    <sequence>/
      frames/<sequence>_<frame-id>.png
      groundtruth/<frame-id>_gt.png
```

The legacy processed layout used by earlier SRRNet experiments is also
accepted directly:

```text
CAD/
  Imgs/
    <sequence>/
      00000.jpg
      00000.png        # optional duplicate; ignored
  GT/
    <sequence>/
      00000.png
```

`CADDataset` pairs files by the final numeric identifier. Original masks with
labels `1/2` and converted masks with values `0/255` are both normalized to
binary class IDs in memory. In the processed layout, RGB input is read only
from `.jpg` files; same-stem `.png` copies in `Imgs/` are ignored.

The paper protocol contains 181 evaluated frames. Inference follows
`configs/datasets/cad_paper_frames.txt` and validates every listed RGB frame
without opening a ground-truth mask. Processed files must be numbered
continuously from `00000` within each sequence and are mapped in order to the
corresponding manifest entries. Predictions retain the canonical manifest
names, and evaluation uses the matching 181 annotations.

## Inference

Inference reads RGB frames in sequence order and saves predictions in this
layout:

```text
<output-directory>/
  <sequence>/
    <source-frame-stem>.png
```

MoCA-Mask inference visits every RGB frame in every discovered test sequence.
CAD inference visits the checked-in 181-frame protocol in manifest order.

## Dataset terms

Datasets are not redistributed by this repository. Review the terms on the
COD10K, MoCA-Mask, and CAD source pages before downloading or publishing
derived artifacts.
