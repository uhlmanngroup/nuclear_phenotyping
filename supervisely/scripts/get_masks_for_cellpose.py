import supervisely_lib as sly

import numpy as np
import os
from PIL import Image
WORKSPACE_ID = 83434
# Initialize API access with your token and server address
api = sly.Api('http://<your-supervisely-instance>', '<your-token>')

# Specify the project to download
project = api.project.get_info_by_name('<your-team>', '<your-project>')

# Download images and annotations
for dataset in api.dataset.get_list(project.id):
    for image in api.image.get_list(dataset.id):
        ann = api.annotation.download(image.id).annotation
        img = api.image.download_np(image.id)

        # Save the image
        image_path = os.path.join('<your-directory>', f'{image.name}.tif')
        Image.fromarray(img).save(image_path)

        # Convert the annotation to a mask and save it
        mask = convert_ann_to_mask(ann)  # Implement this function based on your requirements
        mask_path = os.path.join('<your-directory>', f'{image.name}_masks.tif')
        Image.fromarray(mask).save(mask_path)