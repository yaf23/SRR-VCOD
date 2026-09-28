# SRRNet

[简体中文](README.zh-CN.md) | [English](README.md)

Code for **Scoring, Remember, and Reference: Catching Camouflaged Objects in Videos** (ICCV 2025).

Paper: [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Feng_Scoring_Remember_and_Reference_Catching_Camouflaged_Objects_in_Videos_ICCV_2025_paper.html) · [arXiv:2503.17050](https://arxiv.org/abs/2503.17050)

## Setup

The reference environment is Python 3.8, PyTorch 1.12.1, CUDA 11.3, and `mmcv-full` 1.7.2. Install PyTorch and MMCV for the matching CUDA version, then the remaining packages from `requirements.txt`; see [Installation](docs/INSTALL.md).

Place the downloaded data and weights as described in [Datasets](docs/DATASETS.md) and [Checkpoints](docs/CHECKPOINTS.md).

```text
data/
  COD10K-v3/
  MoCA_Video/
  CamouflagedAnimalDataset/
checkpoints/
  mit_b3.pth
  srrnet_cod10k_iter_50000.pth
  final.pth
```

## Run

The following commands fine-tune, run inference, and evaluate on MoCA-Mask. See [Training configurations](docs/CONFIGS.md) for COD10K pre-training. Dataset paths can also be set with `COD10K_ROOT`, `MOCA_MASK_ROOT`, and `CAD_ROOT`.

```bash
bash tools/dist_train.sh configs/srrnet/moca_finetune.py 3 --work-dir work_dirs/moca_finetune
```

```bash
python infer.py --config configs/srrnet/moca_finetune.py --checkpoint checkpoints/final.pth --data-root data/MoCA_Video --output-dir outputs/moca
```

```bash
python evaluate.py --data-root data/MoCA_Video --pred-dir outputs/moca
```

Inference and evaluation default to MoCA-Mask. For CAD, use `--dataset cad` and the CAD data root. Predictions are saved as `<sequence>/<frame>.png`.

## MoCA-Mask results

`final.pth` was evaluated on 745 frames from 16 test sequences.

| Checkpoint | S-alpha ↑ | Fw-beta ↑ | MAE ↓ | mDice ↑ | mIoU ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| final | 0.726886938 | 0.488703765 | 0.007598855 | 0.512656611 | 0.428250889 |

## Citation

If this code is useful in your research, please cite the paper:

```bibtex
@InProceedings{Feng_2025_VCOD,
    author = {Feng, Yu'ang and Gao, Shuyong and Yan, Fuzhen and Song, Yicheng and Hong, Lingyi and Hu, Junjie and Zhang, Wenqiang},
    title = {Scoring, Remember, and Reference: Catching Camouflaged Objects in Videos},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2025},
    pages = {13043-13052}
}
```
