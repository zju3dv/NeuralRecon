# NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video
### [Project Page](https://zju3dv.github.io/neuralrecon) | [Paper](https://arxiv.org/pdf/2104.00681.pdf)
<br/>

> NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video  
> [Jiaming Sun](https://jiamingsun.ml)<sup>\*</sup>, [Yiming Xie](https://ymingxie.github.io)<sup>\*</sup>, [Linghao Chen](https://github.com/f-sky), [Xiaowei Zhou](http://www.cad.zju.edu.cn/home/xzhou/), [Hujun Bao](http://www.cad.zju.edu.cn/bao/)  
> CVPR 2021 (Oral Presentation and Best Paper Candidate)

<!-- > [NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video](https://arxiv.org/pdf/2104.15838.pdf)   -->
![real-time video](assets/neucon-demo.gif)

<br/>

## TODO List and ETA
- [x] Code (with detailed comments) for training and inference, and the data preparation scripts (2021-5-2).
- [x] Pretrained models on ScanNet (2021-5-2).
- [x] Real-time reconstruction demo on custom ARKit data with instructions (2021-5-7).
- [x] Evaluation code and metrics (expected 2021-6-10).

## How to Use

### Installation
```shell
# Ubuntu 18.04 and above is recommended.
sudo apt install libsparsehash-dev  # you can try to install sparsehash with conda if you don't have sudo privileges.
conda env create -f environment.yaml
conda activate neucon
```
<!-- Follow instructions in [torchsparse](https://github.com/mit-han-lab/torchsparse) to install torchsparse. -->

<details>
  <summary>[FAQ on environment installation]</summary>

 - `AttributeError: module 'torchsparse_backend' has no attribute 'hash_forward'`
   - Clone `torchsparse` to a local directory. If you have done that, recompile and install `torchsparse` after removing the `build` folder.

 - No sudo privileges to install `libsparsehash-dev`
   - Install `sparsehash` in conda (included in `environment.yaml`) and run `export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include` before installing `torchsparse`.

 - For other problems, you can also refer to the [FAQ](https://github.com/mit-han-lab/torchsparse/blob/master/docs/FAQ.md) in `torchsparse`.
</details>

### Pretrained Model on ScanNet
Download the [pretrained weights](https://drive.google.com/file/d/1zKuWqm9weHSm98SZKld1PbEddgLOQkQV/view?usp=sharing) and put it under 
`PROJECT_PATH/checkpoints/release`.
You can also use [gdown](https://github.com/wkentaro/gdown) to download it in command line:
```bash
mkdir checkpoints && cd checkpoints
gdown --id 1zKuWqm9weHSm98SZKld1PbEddgLOQkQV
```

### Real-time Demo on Custom Data with Camera Poses from ARKit.
We provide a real-time demo of NeuralRecon running with self-captured ARKit data.
Please refer to [DEMO.md](DEMO.md) for details.

### Data Preperation for ScanNet
Download and extract ScanNet by following the instructions provided at http://www.scan-net.org/.
<details>
  <summary>[Expected directory structure of ScanNet (click to expand)]</summary>
  
You can obtain the train/val/test split information from [here](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark).
```
DATAROOT
└───scannet
│   └───scans
│   |   └───scene0000_00
│   |       └───color
│   |       │   │   0.jpg
│   |       │   │   1.jpg
│   |       │   │   ...
│   |       │   ...
│   └───scans_test
│   |   └───scene0707_00
│   |       └───color
│   |       │   │   0.jpg
│   |       │   │   1.jpg
│   |       │   │   ...
│   |       │   ...
|   └───scannetv2_test.txt
|   └───scannetv2_train.txt
|   └───scannetv2_val.txt
```
</details>

Next run the data preparation script which parses the raw data format into the processed pickle format.
This script also generates the ground truth TSDFs using TSDF Fusion.  
<details>
  <summary>[Data preparation script]</summary>

```bash
# Change PATH_TO_SCANNET and OUTPUT_PATH accordingly.
# For the training/val split:
python tools/tsdf_fusion/generate_gt.py --data_path PATH_TO_SCANNET --save_name all_tsdf_9 --window_size 9
# For the test split
python tools/tsdf_fusion/generate_gt.py --test --data_path PATH_TO_SCANNET --save_name all_tsdf_9 --window_size 9
```
</details>


### Inference on ScanNet test-set
```bash
python main.py --cfg ./config/test.yaml
```

The reconstructed meshes will be saved to `PROJECT_PATH/results`.


### Evaluation on ScanNet test-set
```
python tools/evaluation.py --model ./results/scene_scannet_release_fusion_eval_47 --n_proc 16
```

Note that `evaluation.py` uses pyrender to render depth maps from the predicted mesh for 2D evaluation.
If you are using headless rendering you must also set the enviroment variable `PYOPENGL_PLATFORM=osmesa`
(see [pyrender](https://pyrender.readthedocs.io/en/latest/install/index.html) for more details).

You can print the results of a previous evaluation run using
```
python tools/visualize_metrics.py --model ./results/scene_scannet_release_fusion_eval_47
```


### Training on ScanNet

Start training by running `./train.sh`.
More info about training (e.g. GPU requirements, convergence time, etc.) to be added soon.
<details>
  <summary>[train.sh]</summary>

```bash
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 main.py --cfg ./config/train.yaml
```
</details>

The training is seperated to two phases and the switching between phases is controlled manually for now:

-  Phase 1 (the first 0-20 epoch), training single fragments.
`MODEL.FUSION.FUSION_ON=False, MODEL.FUSION.FULL=False`


- Phase 2 (the remaining 21-50 epoch), with `GRUFusion`.
`MODEL.FUSION.FUSION_ON=True, MODEL.FUSION.FULL=True`

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{sun2021neucon,
  title={{NeuralRecon}: Real-Time Coherent {3D} Reconstruction from Monocular Video},
  author={Sun, Jiaming and Xie, Yiming and Chen, Linghao and Zhou, Xiaowei and Bao, Hujun},
  journal={CVPR},
  year={2021}
}
```

## Acknowledgment
We would like to specially thank Reviewer 3 for the insightful and constructive comments. We would like to thank Sida Peng , Siyu Zhang and Qi Fang for the proof-reading.
Some of the code in this repo is borrowed from [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch), thanks Xiaoyang!

## Copyright
This work is affiliated with ZJU-SenseTime Joint Lab of 3D Vision, and its intellectual property belongs to SenseTime Group Ltd.

```
Copyright SenseTime. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
