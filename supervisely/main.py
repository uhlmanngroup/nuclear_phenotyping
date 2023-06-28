import supervisely as sly
import os

address = 'https://app.supervise.ly/'
token = os.environ['API_TOKEN']
api = sly.Api(address, token)
project = api.project.get_or_create(workspace_id=82979, name="plast_data")
dataset = api.dataset.get_or_create(project.id, "dataset")
print(project)
# api = sly.Api.from_env()




# from torchvision.datasets import ImageFolder
# from torchvision.transforms import Compose, Normalize
import urllib.request
from glob import glob

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pims
from PIL import Image

data_dir = "data"
import io

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, data, exposure, io
from skimage.color import label2rgb
from skimage.morphology import disk
from skimage.segmentation import mark_boundaries

import bioimage_phenotyping as bip
from bioimage_phenotyping.segmentation import WatershedSegmenter
import matplotlib
matplotlib.use('Agg')

# Overlay the segmentation results on the original image
data_dir = "/home/ctr26/gdrive/+2023_projects/2023_plast_cell/data/plast_cell"
# Lif files are the brightfield images
ext = ".lif"
glob_str = f"{data_dir}/**/*{ext}"
files = glob(glob_str, recursive=True)

# https://github.com/soft-matter/pims/pull/403
pims.bioformats.download_jar(version="6.7.0")

ims = [pims.Bioformats(file) for file in files]



print("ok")
im = pims.Bioformats(files[0])