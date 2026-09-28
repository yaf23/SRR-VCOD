import os
import runpy
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / 'configs' / 'srrnet' / 'cod10k_pretrain.py'


class PretrainRecipeTest(unittest.TestCase):

    def test_cod10k_pretrain_recipe(self):
        checkpoint = '/tmp/mit_b3.pth'
        with mock.patch.dict(
                os.environ, {'SRR_MIT_B3_CHECKPOINT': checkpoint}):
            config = runpy.run_path(str(CONFIG_PATH))

        train = config['data']['train']
        self.assertEqual(config['model']['pretrained'], checkpoint)
        self.assertIsNone(config['load_from'])
        self.assertEqual(config['model']['type'], 'SRRNet')
        self.assertEqual(config['model']['backbone']['type'], 'SRRBackbone')
        self.assertEqual(
            config['model']['backbone']['architecture'],
            'RMATransformerLarge')
        self.assertEqual(config['model']['decode_head']['type'], 'SRRHead')
        self.assertEqual(config['data']['samples_per_gpu'], 4)
        self.assertEqual(config['data']['workers_per_gpu'], 4)
        self.assertEqual(train['type'], 'RepeatDataset')
        self.assertEqual(train['times'], 5)
        self.assertEqual(train['dataset']['type'], 'COD10KDataset')
        self.assertEqual(
            train['dataset']['pipeline'][0]['type'], 'LoadVCODFrames')
        self.assertEqual(
            config['model']['decode_head']['decoder_params']['embed_dim'],
            1024)
        self.assertEqual(config['optimizer']['lr'], 6e-5)
        self.assertEqual(config['lr_config']['policy'], 'poly')
        self.assertEqual(config['lr_config']['warmup_iters'], 2000)
        self.assertEqual(config['runner']['max_iters'], 100000)
        self.assertEqual(config['data']['val']['type'], 'COD10KDataset')
        self.assertEqual(config['data']['val']['split'], 'test')
        self.assertEqual(config['data']['val']['sample_stride'], 10)
        test_transform = config['data']['val']['pipeline'][1]
        self.assertEqual(test_transform['img_scale'], (512, 512))
        self.assertFalse(test_transform['transforms'][0]['keep_ratio'])
        self.assertEqual(config['evaluation']['interval'], 10000)
        self.assertEqual(config['evaluation']['metric'], 'Fwbeta')
        self.assertNotIn('fp16', config)
        self.assertTrue(config['inference_fp16'])


if __name__ == '__main__':
    unittest.main()
