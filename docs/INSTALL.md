# Installation

## Reference environment

The released recipes target this compatibility set:

- Linux
- Python 3.8
- PyTorch 1.12.1
- torchvision 0.13.1
- CUDA 11.3
- `mmcv-full` 1.7.2

An NVIDIA driver compatible with the CUDA 11.3 PyTorch build is required for
GPU training and the reproduced inference result.

## Conda and pip setup

Create a clean environment:

```bash
conda create -n srrnet python=3.8 -y
conda activate srrnet
python -m pip install --upgrade pip
```

Install the CUDA 11.3 PyTorch wheels:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

Install the compiled MMCV wheel that matches CUDA 11.3 and the PyTorch 1.12
binary interface:

```bash
pip install mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
```

The OpenMMLab index is named `torch1.12.0`; its `mmcv-full` 1.7.2 wheels are
the compatible wheels for the PyTorch 1.12 series, including 1.12.1.

Install the remaining packages from the repository root:

```bash
pip install -r requirements.txt
```

The included `mmseg/` runtime is based on
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation); evaluation uses
[PySODMetrics](https://github.com/lartpang/PySODMetrics) ([citation](https://github.com/lartpang/PySODMetrics#citation)).

## Conda file alternative

`environment.yaml` installs Python, PyTorch, torchvision, the CUDA toolkit,
and the packages in `requirements.txt`:

```bash
conda env create -f environment.yaml
conda activate srrnet
pip install mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
```

## Verify the installation

Run:

```bash
python -c "import mmcv, torch; print(torch.__version__, torch.version.cuda, mmcv.__version__)"
python infer.py --help
python evaluate.py --help
python -m tools.train --help
```

The first command should report PyTorch `1.12.1` (the pip wheel includes the
`+cu113` suffix), CUDA `11.3`, and MMCV `1.7.2`.

## Common installation errors

- If MMCV reports missing CUDA operators, uninstall both `mmcv` and
  `mmcv-full`, then reinstall the exact wheel above.
- Do not install MMCV 2.x; this repository uses the MMCV 1.x runner and config
  APIs.
- If pip selects a CPU-only PyTorch build, reinstall PyTorch with the CUDA 11.3
  extra index before installing MMCV.
- Install a current NVIDIA driver when `torch.cuda.is_available()` is false on
  a GPU machine.

After installation, continue with [Datasets](DATASETS.md) and
[Checkpoints](CHECKPOINTS.md).
