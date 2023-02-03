import os
import logging
from copy import deepcopy
from collections import defaultdict

import cv2
import torch
import numpy as np
from torch._C import device
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from ..utils.log import logger, TqdmToLogger, SummaryWriterAvg
from ..model.ps_filters import apply_saturation_filter, get_hue_channel, get_lum_channel, get_sat_channel, rgb_selective_color
from ..utils.misc import save_checkpoint, load_weights
from .optimizer import get_optimizer
from .balanced_dp import BalancedDataParallel

class UpsampleHSLMcTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 image_dump_interval=100,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 freeze=False,
                 net_inputs=('images', 'points')):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset


        

        #logger.info(model)
        self.device = cfg.device
        self.net = model
        self.local_rank = 0
        self._load_weights()

        if freeze:
            for key, value in self.net.named_parameters():
                if 'conv_attention' in key or 'get_refine' in key:
                    continue
                else:
                    value.requires_grad = False
        
        self.optim = get_optimizer(model, optimizer, optimizer_params)
        

        if cfg.multi_gpu:
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            print('local rank', local_rank)
            self.local_rank = local_rank
            torch.cuda.set_device(local_rank)
            cfg.device = torch.device("cuda", local_rank)
            self.device = torch.device('cuda', local_rank)
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net).to(self.device)
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
            self.train_data = torch.utils.data.DataLoader(self.trainset, batch_size=cfg.batch_size, num_workers=cfg.workers, sampler=self.train_sampler)
            self.val_data = torch.utils.data.DataLoader(self.valset, batch_size=cfg.batch_size)
        else:
            self.train_data = DataLoader(
                trainset, cfg.batch_size, shuffle=True,
                drop_last=True, pin_memory=False,
                num_workers=cfg.workers
            )

            self.val_data = DataLoader(
                valset, cfg.val_batch_size, shuffle=False,
                drop_last=True, pin_memory=False,
                num_workers=cfg.workers
            )
            self.net = self.net.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()
        else:
            self.lr_scheduler = None

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        if cfg.input_normalization:
            mean = torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32)
            std = torch.tensor(cfg.input_normalization['std'], dtype=torch.float32)

            self.denormalizator = Normalize((-mean / std), (1.0 / std))
        else:
            self.denormalizator = lambda x: x

    def training(self, epoch):
        if self.sw is None:
            if self.local_rank == 0:
                self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                        flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.multi_gpu:
            self.train_sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        #tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)
        if self.local_rank == 0:
            tbar = tqdm(self.train_data, ncols=150)
        else:
            tbar = self.train_data
        train_loss = 0.0
        lowres_loss = 0.0
        fullres_loss = 0.0
        stage1_l_loss = 0.0
        stage1_s_loss = 0.0
        stage1_h_loss = 0.0
        stage2_l_loss = 0.0
        stage2_s_loss = 0.0
        stage2_h_loss = 0.0
        stage3_l_loss = 0.0
        stage3_s_loss = 0.0
        stage3_h_loss = 0.0
        stage3_debug = 0.0
        stage1_tv_loss = 0.0
        stage2_tv_loss = 0.0
        stage3_tv_loss = 0.0
        stage3_rgb_loss = 0.0
        stage3_rgb_loss_fullres = 0.0

        for metric in self.train_metrics:
            metric.reset_epoch_stats()


        self.net.train()
        use_loss = 1
        for i, batch_data in enumerate(tbar):
            #if self.local_rank == 0:
            #    print(batch_data['image_info'])
            #return 
            global_step = epoch * len(self.train_data) + i
            with torch.autograd.set_detect_anomaly(False):
                loss, losses_logging, splitted_batch_data, outputs = \
                    self.batch_forward(batch_data)
                '''
                if global_step % self.image_dump_interval == 0 and self.local_rank == 0:
                    self.sw.add_image('vis/input_rgb', self.denormalizator(splitted_batch_data['images']), global_step)
                    self.sw.add_image('vis/target_rgb',self.denormalizator(splitted_batch_data['target_images']), global_step)
                    self.sw.add_image('vis/mask', splitted_batch_data['masks'], global_step)
                    self.sw.add_image('vis/stage1_rgb', self.denormalizator(outputs['stage1_output']), global_step)
                    self.sw.add_image('vis/stage2_rgb', self.denormalizator(outputs['stage2_output']), global_step)
                    self.sw.add_image('vis/stage3_rgb', self.denormalizator(outputs['stage3_output']), global_step)
                    self.sw.add_image('vis/pred', self.denormalizator(outputs['images']), global_step)

                    self.sw.add_image('L/mask', splitted_batch_data['masks'], global_step)
                    self.sw.add_image('S/mask', splitted_batch_data['masks'], global_step)
                    self.sw.add_image('H/mask', splitted_batch_data['masks'], global_step)

                    self.sw.add_image('L/stage1_Lmap', outputs['stage1_Lmap'], global_step)
                    self.sw.add_image('S/stage2_Smap', outputs['stage2_Smap'], global_step)
                    self.sw.add_image('H/stage3_Hmap', outputs['stage3_Hmap'], global_step)
                    self.sw.add_image('L/gt_Lmap', splitted_batch_data['gt_Lmap'], global_step)
                    self.sw.add_image('S/gt_Smap', splitted_batch_data['gt_Smap'], global_step)
                    self.sw.add_image('H/gt_Hmap', splitted_batch_data['gt_Hmap'], global_step)
                '''
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            batch_loss = loss.item()
            train_loss += batch_loss
            if self.loss_cfg.get('pixel_loss' + '_weight', 0.0) > 0:
                lowres_loss += losses_logging.get('pixel_loss')[-1]
            if self.loss_cfg.get('pixel_loss_fullres' + '_weight', 0.0) > 0:
                fullres_loss += losses_logging.get('pixel_loss_fullres')[-1]

            if self.loss_cfg.get('stage1_l_loss' + '_weight', 0.0) > 0:
                stage1_l_loss += losses_logging.get('stage1_l_loss')[-1]
            if self.loss_cfg.get('stage2_s_loss' + '_weight', 0.0) > 0:
                stage2_s_loss += losses_logging.get('stage2_s_loss')[-1]
            if self.loss_cfg.get('stage3_h_loss' + '_weight', 0.0) > 0:
                stage3_h_loss += losses_logging.get('stage3_h_loss')[-1]
            if self.loss_cfg.get('stage1_tv_loss' + '_weight', 0.0) > 0:
                stage1_tv_loss += losses_logging.get('stage1_tv_loss')[-1]
            if self.loss_cfg.get('stage2_tv_loss' + '_weight', 0.0) > 0:
                stage2_tv_loss += losses_logging.get('stage2_tv_loss')[-1]
            if self.loss_cfg.get('stage3_tv_loss' + '_weight', 0.0) > 0:
                stage3_tv_loss += losses_logging.get('stage3_tv_loss')[-1]
            if self.loss_cfg.get('stage3_rgb_loss' + '_weight', 0.0) != 0:
                stage3_rgb_loss += losses_logging.get('stage3_rgb_loss')[-1]
            if self.loss_cfg.get('stage3_rgb_loss_fullres' + '_weight', 0.0) != 0:
                stage3_rgb_loss_fullres += losses_logging.get('stage3_rgb_loss_fullres')[-1]

            if self.local_rank == 0:
                tbar.set_description(f'Ep{epoch},L {lowres_loss/(i+1):.4f},H {fullres_loss/(i+1):.4f},' +\
                    f's1_l {stage1_l_loss/(i+1):.4f} ' +\
                    f's2_s {stage2_s_loss/(i+1):.4f} ' + \
                    f's3_h {stage3_h_loss/(i+1):.4f} ' + \
                    #f's1_tv {stage1_tv_loss/(i+1):.4f} ' + \
                    #f's2_tv {stage2_tv_loss/(i+1):.4f} ' + \
                    #f's3_tv {stage3_tv_loss/(i+1):.4f} ' + \
                    f's3_rgb {stage3_rgb_loss/(i+1):.4f} ' + \
                    f's3F_rgb {stage3_rgb_loss_fullres/(i+1):.4f}')

                #if global_step % 50 == 0:
                for loss_name, loss_values in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                    value=loss_values[-1],
                                    global_step=global_step)
                self.sw.add_scalar(tag=f'{log_prefix}Losses/overall',
                                value=batch_loss,
                                global_step=global_step)

        if self.local_rank == 0:
            logger.info(f'Epoch {epoch}, LR {lowres_loss/(len(self.train_data)):.5f}, HR {fullres_loss/(len(self.train_data)):.5f}')

            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                value=np.array(loss_values).mean(),
                                global_step=global_step)
            self.sw.add_scalar(tag=f'{log_prefix}Losses/overall',
                            value=batch_loss,
                            global_step=global_step)

            for k, v in self.loss_cfg.items():
                if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                    v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

            self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                            value=self.lr if self.lr_scheduler is None else self.lr_scheduler.get_lr()[-1],
                            global_step=global_step)

            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/epoch_{metric.name}',
                                value=metric.get_epoch_value(),
                                global_step=epoch, disable_avg=True)

            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=None, multi_gpu=self.cfg.multi_gpu)
            if epoch % self.checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def validation(self, epoch):
        
        if self.sw is None:
            if self.local_rank == 0:
                self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        if self.local_rank == 0:
            tbar = tqdm(self.val_data, ncols=100)
        else:
            tbar = self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        num_batches = 0
        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):

            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            for loss_name, loss_values in batch_losses_logging.items():
                losses_logging[loss_name].extend(loss_values)

            batch_loss = loss.item()
            val_loss += batch_loss
            num_batches += 1

            if self.local_rank == 0:
                tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/num_batches:.6f}')

        if self.local_rank == 0:
            logger.info(f'Epoch {epoch}, loss {val_loss/num_batches:.5f}')

            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/epoch_{metric.name}', value=metric.get_epoch_value(),
                                global_step=epoch, disable_avg=True)
                logger.info(metric.name + '%.3f' % metric.get_epoch_value())
            self.sw.add_scalar(tag=f'{log_prefix}Losses/overall', value=val_loss / num_batches,
                            global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = defaultdict(list)
        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            images, images_fullres, masks = batch_data['images'],batch_data['images_fullres'], batch_data['masks']
            masks_fullres = batch_data['masks_fullres']
            output = self.net(images, images_fullres, masks, masks_fullres)

            for ky, value in output.items():
                if 'image' not in ky:
                    batch_data[ky] = value
            
            target_images = batch_data['target_images']
            if self.loss_cfg.get('stage1_l_loss_weight', 0.0) > 0.0:
                if self.cfg.multi_gpu:
                    batch_data['gt_Lmap'] = self.net.module.model.get_lum(target_images)
                    batch_data['gt_Smap'] = self.net.module.model.get_sat(target_images)
                    batch_data['gt_Hmap'] = self.net.module.model.get_hue(target_images)
                else:
                    batch_data['gt_Lmap'] = self.net.model.get_lum(target_images)
                    batch_data['gt_Smap'] = self.net.model.get_sat(target_images)
                    batch_data['gt_Hmap'] = self.net.model.get_hue(target_images)
            
                output['stage1_Lmap'] = output['stage1_Lmap'] * masks + batch_data['gt_Lmap'] * (1 - masks)
                output['stage2_Smap'] = output['stage2_Smap'] * masks + batch_data['gt_Smap'] * (1 - masks)
                output['stage3_Hmap'] = output['stage3_Hmap'] * masks + batch_data['gt_Hmap'] * (1 - masks)
            #print(output.keys())

            loss = 0.0
            loss = self.add_loss('pixel_loss', loss, losses_logging, validation, output, batch_data)
            loss = self.add_loss('pixel_loss_fullres', loss, losses_logging, validation, output, batch_data)
            
            # supervision on intermediate result
            loss = self.add_loss('stage1_l_loss', loss, losses_logging, validation, output, batch_data)
            loss = self.add_loss('stage2_s_loss', loss, losses_logging, validation, output, batch_data)
            loss = self.add_loss('stage3_h_loss', loss, losses_logging, validation, output, batch_data)
            loss = self.add_loss('stage1_tv_loss', loss, losses_logging, validation, output, batch_data)
            loss = self.add_loss('stage2_tv_loss', loss, losses_logging, validation, output, batch_data)
            loss = self.add_loss('stage3_tv_loss', loss, losses_logging, validation, output, batch_data)
            loss = self.add_loss('stage3_rgb_loss', loss, losses_logging, validation, output, batch_data)
            loss = self.add_loss('stage3_rgb_loss_fullres', loss, losses_logging, validation, output, batch_data)

            #if validation:
            with torch.no_grad():
                for metric in metrics:
                    #print(metric.pred_outputs, metric.gt_outputs)
                    metric.update(
                        *(output.get(x).cpu() for x in metric.pred_outputs),
                        *(batch_data[x].cpu() for x in metric.gt_outputs)
                    )

        return loss, losses_logging, batch_data, output

    def add_loss(self, loss_name, total_loss, losses_logging, validation, net_outputs, batch_data):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        #print(loss_name)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            #print(net_outputs.keys(), batch_data.keys())
            loss = loss_criterion(*(net_outputs.get(x) for x in loss_criterion.pred_outputs),
                                *(batch_data[x] for x in loss_criterion.gt_outputs))
            loss = torch.mean(loss)
            losses_logging[loss_name].append(loss.item())
            loss = loss_weight * loss
            total_loss = total_loss + loss
        elif loss_weight == -1:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*(net_outputs.get(x) for x in loss_criterion.pred_outputs),
                                *(batch_data[x] for x in loss_criterion.gt_outputs))
            loss = torch.mean(loss)
            losses_logging[loss_name].append(loss.item())
            loss = 0 * loss
            total_loss = total_loss + loss
        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(
                str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

        images = splitted_batch_data['images']
        target_images = splitted_batch_data['target_images']
        object_masks = splitted_batch_data['masks']

        image, target_image, object_mask = images[0], target_images[0], object_masks[0, 0]
        image = (self.denormalizator(image).cpu().numpy() * 255).transpose((1, 2, 0))
        target_image = (self.denormalizator(target_image).cpu().numpy() * 255).transpose((1, 2, 0))
        object_mask = np.repeat((object_mask.cpu().numpy() * 255)[:, :, np.newaxis], axis=2, repeats=3)
        predicted_image = (self.denormalizator(outputs['images'].detach()[0]).cpu().numpy() * 255).transpose((1, 2, 0))

        predicted_image = np.clip(predicted_image, 0, 255)

        viz_image = np.hstack((image, object_mask, target_image, predicted_image)).astype(np.uint8)
        _save_image('reconstruction', viz_image[:, :, ::-1])

    def _load_weights(self):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(self.net, self.cfg.weights, verbose=True)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1
            checkpoint_path = checkpoints[0]
            load_weights(self.net, str(checkpoint_path), verbose=True)
        #self.net = self.net.to(self.device)


class _CustomDP(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
