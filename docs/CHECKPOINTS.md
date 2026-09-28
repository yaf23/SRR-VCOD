# Checkpoints

## Download

Download the three files from [Google Drive](https://drive.google.com/drive/folders/1gV4JsXTwFbXYDBrtuysO0eXiUgcsV7Od) and place them in `checkpoints/`.

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `mit_b3.pth` | 178427627 | `92dcdff5d7d4359532ec28d1fb9df1cb4fd56bceffd962dda36eb68489861383` |
| `srrnet_cod10k_iter_50000.pth` | 646303056 | `ee0495bfa5b8413e75ca225c52d603ac0b29d273c88f5ba7028b305dd1d75951` |
| `final.pth` | 646315280 | `05890a90a5c03d01605199f2c92840535d557b50b4d5049b4eb6fbda6339b7ab` |

## Expected files

| Purpose | Default local path | Override |
| --- | --- | --- |
| MiT-B3 backbone initialization | `checkpoints/mit_b3.pth` | `SRR_MIT_B3_CHECKPOINT` |
| COD10K pre-training result used for fine-tuning | `checkpoints/srrnet_cod10k_iter_50000.pth` | `SRR_COD10K_CHECKPOINT` |
| Final MoCA-Mask model used for reported inference | `checkpoints/final.pth` | Pass with `infer.py --checkpoint` |

Recommended local layout:

```text
checkpoints/
  mit_b3.pth
  srrnet_cod10k_iter_50000.pth
  final.pth
```

## Training initialization

COD10K pre-training initializes the MiT-B3 backbone from `mit_b3.pth`.
MiT-B3 is the encoder used by [SegFormer](https://github.com/NVlabs/SegFormer).
Set the checkpoint path with:

```bash
export SRR_MIT_B3_CHECKPOINT=checkpoints/mit_b3.pth
```

MoCA-Mask fine-tuning loads the full COD10K checkpoint:

```bash
export SRR_COD10K_CHECKPOINT=checkpoints/srrnet_cod10k_iter_50000.pth
```

Use `--load-from` to replace the config's full-model checkpoint. Full-model
checkpoints are loaded strictly. Use `--resume-from` only to resume a stopped
run with optimizer and runner state.

## Inference checkpoint

Inference disables config-level backbone initialization and strictly loads the
checkpoint supplied on the command line:

```bash
python infer.py \
  --config configs/srrnet/moca_finetune.py \
  --checkpoint checkpoints/final.pth \
  --dataset cad \
  --data-root data/CamouflagedAnimalDataset \
  --output-dir outputs/cad
```

Place the final model at `checkpoints/final.pth`, or pass its location with
`--checkpoint`.
