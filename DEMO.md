
# Real-time Demo Tutorial (WIP)

## Pose calculation with COLMAP?
#### Demo (your own dataset)
The directory structure should look like:
```
DATAROOT
└───fragments.pkl
└───images
│   └───0.jpg
│   └───1.jpg
│   |   ...
```
The structure of fragment.pkl look like:
```
[
{'scene': scene_name: [str],
'fragment_id': fragment id: [int],
'image_ids': image id: [int],
'extrinsics': poses: [metrics:4X4],
'intrinsics': intrinsics: [metrics: 3X3]
}
...
]
```
([example](tools/ios_logger_process.py) for generating fragment.pkl)

Test NeuralRecon on DemoDataset: 
```shell
python demo.py --cfg ./config/demo.yaml
```
You can change your dataset path in [demo.yaml](config/demo.yaml)