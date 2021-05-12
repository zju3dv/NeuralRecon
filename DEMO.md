
# Real-time Demo with Custom ARKit Data
In this tutorial we introduce the real-time demo of NeuralRecon running with self-captured ARKit data. If you don't want to take the effort capturing your own data, you can download the [example data](https://drive.google.com/file/d/1i7yDY-oautHgwrDPUrB7BuZytiIYhxfY/view?usp=sharing) and skip step 1.

To capture data and run this demo, an Apple device (iPhone or iPad) with ARKit support is required. 
Generally speaking, devices released after 2017 (e.g. iPhone 7 and later generations) are all supported. 
You can search for 'arkit' on [this page](https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/DeviceCompatibilityMatrix/DeviceCompatibilityMatrix.html) to find out.
You will also need a Mac computer to compile the data capture app and a GPU-enabled machine (GPU memory > 2GB) to run NeuralRecon.

## Step 1: Capture video data with camera poses from ARKit
For now we use [ios_logger](https://github.com/Varvrar/ios_logger) as the capture app, and you will have to compile it yourself.
We are making an attempt to release a new capture app that is available to download from the App Store. 


### Compile ios_logger

1. Download and install [Xcode](https://apps.apple.com/us/app/xcode/id497799835?mt=12).
2. `git clone https://github.com/Varvrar/ios_logger`
3. Follow [this tutorial](https://ioscodesigning.com/generating-code-signing-files) to generate [a certificate](https://ioscodesigning.com/generating-code-signing-files/#generate-a-code-signing-certificate-using-xcode) and [a provisioning profile](https://ioscodesigning.com/generating-code-signing-files/#generate-a-provisioning-profile-with-xcode). (Don't be scared if you find this tutorial is very long and complex😉, it's actually quite simple with Xcode automatically handling most of the work.)
4. Follow the [README](https://github.com/Varvrar/ios_logger#build-and-run) of ios_logger to run it on your device.

### Capture the data
You can follow [these steps](https://github.com/Varvrar/ios_logger#collect-datasets) to capture the data. 
Although a clean indoor environment is prefered since it's closer to the training dataset ScanNet, most scenarios (indoor and outdoor) should work just fine.
Be sure to move around your device frequently during capture to get more views with covisibility on the same place.

## Step 2: Run the demo
After [retrieving the captured data](https://github.com/Varvrar/ios_logger#get-saved-datasets) and transfer it to a GPU-enabled machine, you are good to proceed. Notice that it's a good idea to start with the [example data](https://drive.google.com/file/d/1i7yDY-oautHgwrDPUrB7BuZytiIYhxfY/view?usp=sharing) to make sure the environment for NeuralRecon is correctly configured.

1. Change the data path in [demo.yaml](config/demo.yaml).
2. Optionally, you can enable the `VIS_INCREMENTAL` flag to get a real-time visualization during reconstruction if you are on a local machine. `SAVE_INCREMENTAL` saves the incremental meshes at each step.
3. Optionally, you can disable the `REDUCE_GPU_MEM` flag to get the maximum inference speed (~3 keyframs/sec speed-up).
4. Run NeuralRecon demo: 
`
python demo.py --cfg ./config/demo.yaml
`

The reconstructed mesh will be available under `results/scene_demo_checkpoints_fusion_eval_47`. 
You can open the ply file with [MeshLab](https://www.meshlab.net/).

## Illustrations to the processed data format from ios_logger
For those who are interested to reuse the captured data for other projects:

The directory structure:
```
DATAROOT
└───fragments.pkl
└───images
│   └───0.jpg
│   └───1.jpg
│   |   ...
```
The structure of `fragments.pkl`:
```
[
{'scene': scene_name: [str],
'fragment_id': fragment id: [int],
'image_ids': image id: [int],
'extrinsics': poses: [matrix: 4X4],
'intrinsics': intrinsics: [matrix: 3X3]
}
...
]
```

[Here](tools/process_arkit_data.py) is the code for generating `fragments.pkl`.
