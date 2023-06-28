# %%
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from splinedist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from splinedist.matching import matching, matching_dataset
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D, StarDist2D

np.random.seed(42)
lbl_cmap = random_label_cmap()

import splinegenerator as sg
from splinedist.utils import phi_generator, grid_generator, get_contoursize_max
# from stardist.models import StarDist2D

# from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
from splinedist import fill_label_holes

from splinedist import random_label_cmap, _draw_polygons, export_imagej_rois
from splinedist.models import SplineDist2D
from splinedist.utils import iou_objectwise, iou 

np.random.seed(6)
lbl_cmap = random_label_cmap()
import tensorflow as tf
tf.config.experimental.set_memory_growth  = True

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--image_in','--image_in',type=str, help='images_dir')
parser.add_argument('--figure_out','--figure_out',type=str, help='figure_out')
parser.add_argument('--model_path','--model_path',default="models",type=str, help='model')


args = parser.parse_args()

image_in = args.image_in
model_path = args.model_path
figure_out = args.figure_out

print(args)

X = [image_in]
X = list(map(imread,X))
plt.imshow(X[0])
plt.show()

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

model = SplineDist2D(None, name='splinedist', basedir=model_path)
model = StarDist2D(None, name='splinedist', basedir=model_path)

img = normalize(X[0], 1,99.8, axis=axis_norm)
labels, details = model.predict_instances(img)

plt.figure(figsize=(8,8))
plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
plt.axis('off')
plt.savefig(figure_out)



