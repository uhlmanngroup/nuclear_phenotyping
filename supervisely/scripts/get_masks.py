WORKSPACE_ID = 83434

import supervisely_lib as sly
import json
import requests
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
# Initialize API access with your token and server address
# api = sly.Api('http://<your-supervisely-instance>', '<your-token>')
address = "https://app.supervise.ly/"
token = os.environ["API_TOKEN"]
api = sly.Api(address, token)
# Specify the project to download
# project = api.project.get_info_by_name('<your-team>', '<your-project>')
project = api.project.get_or_create(
    workspace_id=WORKSPACE_ID, name="plast_data"
)
import numpy as np
import cv2, zlib, base64, io
from PIL import Image

def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0,0,0,255,255,255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')


out_dir = "data/annotated"
# Download images and annotations
for dataset in api.dataset.get_list(project.id):
    for image in api.image.get_list(dataset.id):
        image_id = image.id
        image_id = 291426010
        ann = api.annotation.download(image.id).annotation
        # img = api.image.download_np(image.id)
        
        if ann["objects"] != []:
            print(ann)
            # ann["objects"][0]["bitmap"]
            # ann["objects"][0]["bitmap"]["data"]
            img = api.image.download_np(image.id)
            mask = base64_2_mask(ann["objects"][0]["bitmap"]["data"])
            name = "20230216/NMuMG-mut218_5um_20230213_useless/t=104_z=0_c=1"
            api.image.get_info_by_name(dataset.id,name=name).id
            # plt.imshow(mask)


        # Save the image
        # Image.fromarray(img).save(f'{image.name}.jpg')
        # Save the image
        # image_path = os.path.join(out_dir, f'{image.name}.tif')
        # Image.fromarray(img).save(image_path)

        # Convert the annotation to a mask and save it
        # mask = convert_ann_to_mask(ann)  # Implement this function based on your requirements
        # mask_path = os.path.join(out_dir, f'{image.name}_masks.tif')
        # Image.fromarray(mask).save(mask_path)
        # Convert the annotation to COCO format and save it
        # coco_ann = convert_to_coco(ann)  # Implement this function based on your requirements
        # with open(f'{image.name}.json', 'w') as f:
            # json.dump(coco_ann, f)

# # Now, you can use the COCO dataset loader from PyTorch to load the data
# coco_data = datasets.CocoDetection(
#     root='.',  # Specify the root directory where your images and annotations are saved
#     annFile='.',  # Specify the directory where your annotations are saved
#     transform=transforms.ToTensor(),
# )

# # Create a DataLoader
# data_loader = DataLoader(coco_data, batch_size=32, shuffle=True)