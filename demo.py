import argparse
import os, time

import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from models import NeuralRecon
from utils import SaveScene
from config import cfg, update_config
from datasets import find_dataset_def, transforms
from tools.process_arkit_data import process_data


parser = argparse.ArgumentParser(description='NeuralRecon Real-time Demo')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)

parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)

# parse arguments and check
args = parser.parse_args()
update_config(cfg, args)

if not os.path.exists(os.path.join(cfg.TEST.PATH, 'SyncedPoses.txt')):
    logger.info("First run on this captured data, start the pre-processing...")
    process_data(cfg.TEST.PATH)
else:
    logger.info("Found SyncedPoses.txt, skipping data pre-processing...")

logger.info("Running NeuralRecon...")
transform = [transforms.ResizeImage((640, 480)),
             transforms.ToTensor(),
             transforms.RandomTransformSpace(
                 cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation=False, random_translation=False,
                 paddingXY=0, paddingZ=0, max_epoch=cfg.TRAIN.EPOCHS),
             transforms.IntrinsicsPoseToProjection(cfg.TEST.N_VIEWS, 4)]
transforms = transforms.Compose(transform)
ARKitDataset = find_dataset_def(cfg.DATASET)
test_dataset = ARKitDataset(cfg.TEST.PATH, "test", transforms, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)
data_loader = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TEST.N_WORKERS, drop_last=False)

# model
logger.info("Initializing the model on GPU...")
model = NeuralRecon(cfg).cuda().eval()
model = torch.nn.DataParallel(model, device_ids=[0])

# use the latest checkpoint file
saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
loadckpt = os.path.join(cfg.LOGDIR, saved_models[-1])
logger.info("Resuming from " + str(loadckpt))
state_dict = torch.load(loadckpt)
model.load_state_dict(state_dict['model'], strict=False)
epoch_idx = state_dict['epoch']
save_mesh_scene = SaveScene(cfg)

logger.info("Start inference..")
duration = 0.
gpu_mem_usage = []
frag_len = len(data_loader)
with torch.no_grad():
    for frag_idx, sample in enumerate(tqdm(data_loader)):
        # save mesh if: 1. SAVE_SCENE_MESH and is the last fragment, or
        #               2. SAVE_INCREMENTAL, or
        #               3. VIS_INCREMENTAL
        save_scene = (cfg.SAVE_SCENE_MESH and frag_idx == frag_len - 1) or cfg.SAVE_INCREMENTAL or cfg.VIS_INCREMENTAL

        start_time = time.time()
        outputs, loss_dict = model(sample, save_scene)
        duration += time.time() - start_time

        if cfg.REDUCE_GPU_MEM:
            # will slow down the inference
            torch.cuda.empty_cache()

        # vis or save incremental result.
        scene = sample['scene'][0]
        save_mesh_scene.keyframe_id = frag_idx
        save_mesh_scene.scene_name = scene.replace('/', '-')

        if cfg.SAVE_INCREMENTAL:
            save_mesh_scene.save_incremental(epoch_idx, 0, sample['imgs'][0], outputs)

        if cfg.VIS_INCREMENTAL:
            save_mesh_scene.vis_incremental(epoch_idx, 0, sample['imgs'][0], outputs)

        if cfg.SAVE_SCENE_MESH and frag_idx == frag_len - 1:
            assert 'scene_tsdf' in outputs, \
            """Reconstruction failed. Potential reasons could be:
                1. Wrong camera poses.
                2. Extremely difficult scene.
                If you can run with the demo data without any problem, please submit a issue with the failed data attatched, thanks!
            """
            save_mesh_scene.save_scene_eval(epoch_idx, outputs)
        
        gpu_mem_usage.append(torch.cuda.memory_reserved())
        
summary_text = f"""
Summary:
    Total number of fragments: {frag_len} 
    Average keyframes/sec: {1 / (duration / (frag_len * cfg.TEST.N_VIEWS))}
    Average GPU memory usage (GB): {sum(gpu_mem_usage) / len(gpu_mem_usage) / (1024 ** 3)} 
    Max GPU memory usage (GB): {max(gpu_mem_usage) / (1024 ** 3)} 
"""
print(summary_text)

if cfg.VIS_INCREMENTAL:
    save_mesh_scene.close()