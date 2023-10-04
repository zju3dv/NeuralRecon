import os
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import requests
import torch
from tqdm import tqdm
import glob

# Initialize the model and processor, ensuring GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
model.to(device)

# Path to the scene folders
scenes_folder = "/mnt/sdd/scannet_subset/scans"

# Iterate through the scene folders
for scene_folder in tqdm(sorted(glob.glob(os.path.join(scenes_folder, "*")))):
    if not os.path.isdir(scene_folder):
        continue
    
    # Get the path to the color folder
    color_folder = os.path.join(scene_folder, "color")

    if not os.path.exists(color_folder):
        continue
    
    # Create a 'seg' folder
    seg_folder = os.path.join(scene_folder, "seg")
    os.makedirs(seg_folder, exist_ok=True)

    for image_path in tqdm(glob.glob(os.path.join(color_folder, "*"))):
        try:
            # Open the image
            image = Image.open(image_path)
            inputs = image_processor(image, return_tensors="pt")

            # Send the inputs to the GPU
            inputs.to(device)

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process to get the instance segmentation map
            pred_instance_map = image_processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]

            # Move the output back to CPU, convert to numpy, and change data type to uint8
            pred_instance_map = pred_instance_map.to("cpu").numpy().astype('uint8')
            pred_instance_map = Image.fromarray(pred_instance_map)

            # Save the segmentation map
            seg_path = os.path.join(seg_folder, os.path.basename(image_path))
            pred_instance_map.save(seg_path)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
