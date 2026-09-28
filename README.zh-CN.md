# SRRNet

[简体中文](README.zh-CN.md) | [English](README.md)

**Scoring, Remember, and Reference: Catching Camouflaged Objects in Videos**（ICCV 2025）的论文代码。

论文：[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Feng_Scoring_Remember_and_Reference_Catching_Camouflaged_Objects_in_Videos_ICCV_2025_paper.html) · [arXiv:2503.17050](https://arxiv.org/abs/2503.17050)

## 准备

参考环境为 Python 3.8、PyTorch 1.12.1、CUDA 11.3 和 `mmcv-full` 1.7.2。先安装与 CUDA 匹配的 PyTorch 和 MMCV，再通过 `requirements.txt` 安装其余依赖；详见 [安装说明](docs/INSTALL.md)。

按照 [数据集说明](docs/DATASETS.md) 和 [权重说明](docs/CHECKPOINTS.md) 放置下载的数据及模型权重。

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

## 运行

以下命令以 MoCA-Mask 为例，依次进行微调、推理和评估。COD10K 预训练见 [训练配置](docs/CONFIGS.md)。也可通过 `COD10K_ROOT`、`MOCA_MASK_ROOT`、`CAD_ROOT` 指定数据集路径。

```bash
bash tools/dist_train.sh configs/srrnet/moca_finetune.py 3 --work-dir work_dirs/moca_finetune
```

```bash
python infer.py --config configs/srrnet/moca_finetune.py --checkpoint checkpoints/final.pth --data-root data/MoCA_Video --output-dir outputs/moca
```

```bash
python evaluate.py --data-root data/MoCA_Video --pred-dir outputs/moca
```

推理和评估默认使用 MoCA-Mask。处理 CAD 时添加 `--dataset cad` 并指定 CAD 数据目录。预测结果保存为 `<序列>/<帧>.png`。

## MoCA-Mask 结果

`final.pth` 在 16 个测试序列的 745 帧上完成评估。

| 权重 | S-alpha ↑ | Fw-beta ↑ | MAE ↓ | mDice ↑ | mIoU ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| final | 0.726886938 | 0.488703765 | 0.007598855 | 0.512656611 | 0.428250889 |

## 引用

如果本项目对你的研究有帮助，请引用论文：

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
