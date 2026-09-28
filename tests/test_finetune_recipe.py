import os
import runpy
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / 'configs' / 'srrnet' / 'moca_finetune.py'


class FinetuneRecipeTest(unittest.TestCase):

    def test_moca_finetune_recipe(self):
        checkpoint = '/tmp/cod10k_iter_50000.pth'
        with mock.patch.dict(
                os.environ, {'SRR_COD10K_CHECKPOINT': checkpoint}):
            config = runpy.run_path(str(CONFIG_PATH))

        self.assertEqual(config['load_from'], checkpoint)
        self.assertIsNone(config['model']['pretrained'])
        self.assertEqual(config['model']['type'], 'SRRNet')
        self.assertEqual(config['model']['backbone']['type'], 'SRRBackbone')
        self.assertEqual(
            config['model']['backbone']['architecture'],
            'RMATransformerLarge')
        self.assertEqual(config['model']['decode_head']['type'], 'SRRHead')
        self.assertEqual(config['data']['samples_per_gpu'], 4)
        self.assertEqual(config['data']['workers_per_gpu'], 4)
        self.assertEqual(config['data']['train']['type'], 'MoCAMaskDataset')
        self.assertEqual(
            config['data']['train']['pipeline'][0]['type'], 'LoadVCODFrames')
        self.assertEqual(config['data']['val']['type'], 'CADDataset')
        self.assertTrue(config['data']['val']['paper_protocol'])
        self.assertEqual(config['data']['test']['type'], 'MoCAMaskDataset')
        self.assertEqual(config['data']['test']['split'], 'test')
        self.assertEqual(
            config['model']['decode_head']['decoder_params']['embed_dim'],
            1024)
        self.assertEqual(config['optimizer']['lr'], 1e-5)
        self.assertEqual(config['lr_config']['policy'], 'fixed')
        self.assertEqual(config['runner']['max_iters'], 10000)
        self.assertEqual(config['evaluation']['interval'], 2000)
        self.assertEqual(config['evaluation']['metric'], 'Fwbeta')
        self.assertNotIn('fp16', config)
        self.assertTrue(config['inference_fp16'])


if __name__ == '__main__':
    unittest.main()
