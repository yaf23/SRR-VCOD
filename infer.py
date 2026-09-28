"""Public command-line entry point for SRRNet sequence inference."""

import argparse

from mmseg.apis.inference import run_inference


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Run single-pass SRRNet inference on downloaded MoCA-Mask or '
            'CAD video frames.'))
    parser.add_argument(
        '--config', required=True, help='Path to the SRRNet model config.')
    parser.add_argument(
        '--checkpoint', required=True, help='Path to the model checkpoint.')
    parser.add_argument(
        '--data-root', required=True,
        help='Root of the downloaded MoCA-Mask or CAD dataset.')
    parser.add_argument(
        '--dataset', default='moca', type=str.lower,
        choices=('moca', 'cad'),
        help='Downloaded dataset layout to index (default: moca).')
    parser.add_argument(
        '--output-dir', required=True,
        help='Directory in which binary prediction masks are written.')
    parser.add_argument(
        '--device', default='cuda:0',
        help='PyTorch device, for example cuda:0 or cpu (default: cuda:0).')
    parser.add_argument(
        '--scores-json', default=None,
        help=(
            'Optional JSON file for predicted error scores and reference '
            'updates. No score file is written when omitted.'))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run_inference(args)


if __name__ == '__main__':
    main()
