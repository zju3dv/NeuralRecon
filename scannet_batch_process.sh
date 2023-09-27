#!/bin/bash

input_dir="/mnt/sdd/scannet/scans"
output_dir="/mnt/sdd/scannet/scans"


for scene_dir in "$input_dir"/scene*; do
    if [ -d "$scene_dir" ]; then
        scene_name=$(basename "$scene_dir")

        echo "Processing scene: $scene_dir/$scene_name"

        python /mnt/sdd/ScanNet/SensReader/python/reader.py --filename "$scene_dir/$scene_name.sens" \
                         --output_path "$output_dir/$scene_name/" \
                         --export_color_images \
                         --export_depth_images \
                         --export_poses \
                         --export_intrinsics
    fi
done