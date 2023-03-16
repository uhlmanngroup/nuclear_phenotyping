from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
from splinedist import fill_label_holes

from splinedist import random_label_cmap, _draw_polygons, export_imagej_rois
from splinedist.models import SplineDist2D
from splinedist.utils import iou_objectwise, iou 

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# tf.keras.backend.set_session(tf.Session(config=config));

np.random.seed(6)
lbl_cmap = random_label_cmap()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import os
# os.environ['CUDA_VISIBLE_DEVICES']="0"
# %config Completer.use_jedi = False

from tqdm import tqdm
from PIL import Image
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--image_in','--image_in',type=str, help='images_dir')
parser.add_argument('--figure_out','--figure_out',type=str, help='figure_out')
parser.add_argument('--model_path','--model_path',default="models",type=str, help='model')
parser.add_argument('--model_name','--model_name',default="model_name",type=str, help='model_name')
parser.add_argument('--instance','--instance',type=str, help='instance')
parser.add_argument('--control_points','--control_points',type=str, help='control_points')
parser.add_argument('--raw_image','--raw_image',type=str, help='raw_image')
args = parser.parse_args()

image_in = args.image_in
model_path = args.model_path
model_name = args.model_name
figure_out = args.figure_out
instance = args.instance
control_points = args.control_points
raw_image = args.raw_image

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
    
plt.imshow(X[0])
plt.show()


# show all test images
if False:
    fig, ax = plt.subplots(7,8, figsize=(16,16))
    for i,(a,x) in enumerate(zip(ax.flat, X)):
        a.imshow(x if x.ndim==2 else x[...,0], cmap='gray')
        a.set_title(i)
    [a.axis('off') for a in ax.flat]
    plt.tight_layout()



model = SplineDist2D(None, name=model_name, basedir=model_path)

image_dims = (2048, 2048)

im = X[0]
pad_width = (np.subtract(image_dims, im.shape))/2
pad_width_x, pad_width_y = pad_width
pad_vector = np.rint([[np.floor(pad_width_x),
                    np.ceil(pad_width_x)],
                    [np.floor(pad_width_y),
                    np.ceil(pad_width_y)]]).astype(int)
im_padded = np.pad(im,
                pad_width=pad_vector,
                mode='constant',
                constant_values=((0, 0), (0, 0)))

img = normalize(im_padded, 1,99.8, axis=axis_norm)
labels, details = model.predict_instances(img)


# plt.figure(figsize=(16,16))
# plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1))
# plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
# plt.axis('off');


# masks = []
# controlpoints = []

# # for i in tqdm(range(len(X))):
# img = normalize(X[0], 1,99.8, axis=axis_norm)
# labels, details = model.predict_instances(img)

coord = details['coord']


def center(pts):
    c = [sum(pts[:,0])/len(pts),sum(pts[:,1])/len(pts)]
    return np.array([[p[0]-c[0],p[1]-c[1]] for p in pts])

#translate the centroids of the objects to the origin
def translate(coord): 
    for i in range(len(coord)):
        object_coefs = coord[i]
        object_coefs = np.transpose(object_coefs, (1,0))
        centroid = [sum(object_coefs[:,0])/len(object_coefs),sum(object_coefs[:,1])/len(object_coefs)]
        object_coefs_translated = np.array([[p[0]-centroid[0],p[1]-centroid[1]] for p in object_coefs])
        return object_coefs_translated 

# %%
# def example(model, i, show_dist=True):
#     img = normalize(X[i], 1,99.8, axis=axis_norm)
#     labels, details = model.predict_instances(img)

#     plt.figure(figsize=(13,10))
#     img_show = img if img.ndim==2 else img[...,0]
#     coord, points, prob = details['coord'], details['points'], details['prob']
#     print(coord.shape, points.shape, prob.shape)
#     plt.subplot(121); plt.imshow(img_show, cmap='gray'); plt.axis('off')
#     a = plt.axis()
#     _draw_polygons(coord, points, prob, show_dist=show_dist)
#     plt.axis(a)
#     plt.subplot(122); plt.imshow(img_show, cmap='gray'); plt.axis('off')
#     plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
#     plt.tight_layout()
#     plt.show()

# cellesce_results = [X_ids, controlpoints]
# cellesce_results = np.array(cellesce_results, dtype = 'object')
from skimage import io

io.imsave(instance,labels)
io.imsave(raw_image,im_padded)

# print(coord)
# print(coord.shape)

# mask_num,y,x = X.shape
# in your case
# a,b,c = 1797, 500
# print(pd.DataFrame.from_records(coord))
mask_num,y,x = coord.shape

df = pd.DataFrame(
    data=coord.flatten(),
    index=pd.MultiIndex.from_product(
            [np.arange(0,mask_num), np.arange(0,y),np.arange(0,x)],
            names=["mask_num","y","x"],
            ),
    columns=["Value"]
)
# print(df)

# np.save(control_points,coord)
df.to_csv(control_points)
