"""Minimal segmentor interface used by SRRNet and the MMCV runner."""

import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16


class BaseSegmentor(nn.Module, metaclass=ABCMeta):
    """Base class for segmentors."""

    def __init__(self):
        super().__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """Whether the segmentor has a neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """Whether the segmentor has an auxiliary head."""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """Whether the segmentor has a decode head."""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from input images."""
        raise NotImplementedError

    @abstractmethod
    def encode_decode(self, img, img_metas):
        """Encode images with the backbone and decode them into a
        semantic segmentation map of the same size as input."""
        raise NotImplementedError

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """Run the training forward pass."""
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        """Run single-augmentation inference."""
        raise NotImplementedError

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Run multi-augmentation inference."""
        raise NotImplementedError

    def init_weights(self, pretrained=None):
        """Initialize the weights in segmentor.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('Load model from: %s', pretrained)

    def forward_test(self, imgs, img_metas, **kwargs):
        """Dispatch single- or multi-augmentation inference.

        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                f'Number of augmentations ({len(imgs)}) does not match '
                f'number of metadata groups ({len(img_metas)}).')

        for img_meta in img_metas:
            image_shapes = [meta['img_shape'] for meta in img_meta]
            if not all(shape == image_shapes[0] for shape in image_shapes):
                raise ValueError(
                    'Images in one augmentation batch must share img_shape.')
            padded_shapes = [meta['pad_shape'] for meta in img_meta]
            if not all(shape == padded_shapes[0]
                       for shape in padded_shapes):
                raise ValueError(
                    'Images in one augmentation batch must share pad_shape.')

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``return_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of the data loader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.detach().clone()
                loss_value.div_(dist.get_world_size())
                dist.all_reduce(loss_value)
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
