# Training configurations

Run commands from the repository root. The final number is the GPU count;
`--work-dir` selects where logs and checkpoints are saved.

| Config | Training data | Initial weights | Validation | Schedule |
| --- | --- | --- | --- | --- |
| `configs/srrnet/cod10k_pretrain.py` | COD10K Train | MiT-B3 backbone | COD10K Test subset, every 10k iterations | 100k iterations |
| `configs/srrnet/moca_finetune.py` | MoCA-Mask Train | Full COD10K model | CAD, every 2k iterations | 10k iterations |

```bash
bash tools/dist_train.sh configs/srrnet/cod10k_pretrain.py 3 --work-dir work_dirs/cod10k_pretrain
bash tools/dist_train.sh configs/srrnet/moca_finetune.py 3 --work-dir work_dirs/moca_finetune
```

The configs use `data/` and `checkpoints/` by default. See [Datasets](DATASETS.md)
for data paths and [Checkpoints](CHECKPOINTS.md) for initialization weights.
Fine-tuning requires the COD10K checkpoint produced by pre-training.
