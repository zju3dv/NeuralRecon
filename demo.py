import argparse
import os
import gc
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from loguru import logger

from utils import tensor2float, DictAverageMeter, SaveScene, make_nograd_func
from models import NeuralRecon
from datasets import find_dataset_def, transforms
from datasets.sampler import DistributedSampler
from config import cfg, update_config
from ops.comm import *


def args():
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of NeuralRecon')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    # parse arguments and check
    args = parser.parse_args()

    return args


args = args()
update_config(cfg, args)

cfg.defrost()
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
logger.info('number of gpus: {}'.format(num_gpus))
cfg.DISTRIBUTED = num_gpus > 1

if cfg.DISTRIBUTED:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()
cfg.LOCAL_RANK = args.local_rank
cfg.freeze()

torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)

# create logger
if is_main_process():
    if not os.path.isdir(cfg.LOGDIR):
        os.makedirs(cfg.LOGDIR)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger.info("current time", current_time_str)
    logfile_path = os.path.join(cfg.LOGDIR, f'{current_time_str}_{cfg.MODE}.log')
    print('creating log file', logfile_path)
    logger.add(logfile_path, format="{time} {level} {message}", level="INFO")

# Augmentation
n_views = cfg.TEST.N_VIEWS
random_rotation = False
random_translation = False
paddingXY = 0
paddingZ = 0

transform = []
transform += [transforms.ResizeImage((640, 480)),
              transforms.ToTensor(),
              transforms.RandomTransformSpace(
                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation, random_translation,
                  paddingXY, paddingZ, max_epoch=cfg.TRAIN.EPOCHS),
              transforms.IntrinsicsPoseToProjection(n_views, 4),
              ]

transforms = transforms.Compose(transform)

MVSDataset = find_dataset_def(cfg.DATASET)
test_dataset = MVSDataset(cfg.TEST.PATH, "test", transforms, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)

if cfg.DISTRIBUTED:
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    TestImgLoader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=test_sampler,
        num_workers=cfg.TEST.N_WORKERS,
        pin_memory=True,
        drop_last=False
    )
else:
    TestImgLoader = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TEST.N_WORKERS,
                               drop_last=False, )

# model, optimizer
model = NeuralRecon(cfg)
if cfg.DISTRIBUTED:
    model.cuda()
    model = DistributedDataParallel(
        model, device_ids=[cfg.LOCAL_RANK], output_device=cfg.LOCAL_RANK,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=False,
        find_unused_parameters=True
    )
else:
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=(0.9, 0.999), weight_decay=cfg.TRAIN.WD)


def test(from_latest=False):
    ckpt_list = []
    saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if from_latest:
        saved_models = saved_models[-1:]
    for ckpt in saved_models:
        if ckpt not in ckpt_list:
            # use the latest checkpoint file
            loadckpt = os.path.join(cfg.LOGDIR, ckpt)
            logger.info("resuming " + str(loadckpt))
            state_dict = torch.load(loadckpt)
            model.load_state_dict(state_dict['model'], strict=False)
            optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
            optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
            epoch_idx = state_dict['epoch']

            TestImgLoader.dataset.tsdf_cashe = {}

            avg_test_scalars = DictAverageMeter()
            save_mesh_scene = SaveScene(cfg)
            for batch_idx, sample in enumerate(TestImgLoader):
                for n in sample['fragment']:
                    logger.info(n)
                start_time = time.time()
                loss, scalar_outputs, outputs = test_sample(sample)
                logger.info('Epoch {}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, batch_idx,
                                                                                            len(TestImgLoader), loss,
                                                                                            time.time() - start_time))
                scalar_outputs.update({'time': time.time() - start_time})
                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs

                if batch_idx % 100 == 0:
                    logger.info("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader),
                                                                       avg_test_scalars.mean()))

                # save mesh
                if cfg.SAVE_SCENE_MESH or cfg.SAVE_INCREMENTAL:
                    save_mesh_scene(outputs, sample, epoch_idx)
            logger.info("epoch {} avg_test_scalars:".format(epoch_idx), avg_test_scalars.mean())

            ckpt_list.append(ckpt)

    time.sleep(10)


@make_nograd_func
def test_sample(sample):
    model.eval()

    outputs, loss_dict = model(sample)
    loss = loss_dict['total_loss']

    return tensor2float(loss), tensor2float(loss_dict), outputs


if __name__ == '__main__':
    assert cfg.MODE == 'test'
    test()
