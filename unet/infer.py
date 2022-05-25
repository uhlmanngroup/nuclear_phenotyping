# # %%
# from unet_nuclei import *
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.io import imread
# import imageio
# import os
# import glob
# from tqdm import tqdm
# from random import shuffle

# # from PIL import Image
# # %matplotlib inline

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# %%
import unet_nuclei
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import glob
from tqdm import tqdm
from random import shuffle

# from PIL import Image
# %matplotlib inline

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# %%

# os.environ["KERAS_BACKEND"] =  "tensorflow"
# os.environ["KERAS_BACKEND"] =  "cntk"

# %%

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--image_in','--image_in',type=str, help='images_dir')
parser.add_argument('--figure_out','--figure_out',type=str, help='figure_out')
parser.add_argument('--labels_out','--labels_out',default="labels_out",type=str, help='labels_out')

parser.add_argument('--background_image','--background',default="background",type=str, help='background')
parser.add_argument('--foreground_image','--foreground_image',default="foreground_image",type=str, help='foreground_image')
parser.add_argument('--boundary_image','--boundary',default="boundary",type=str, help='boundary')
parser.add_argument('--raw_image','--raw_image',default="raw_image",type=str, help='raw_image')


args = parser.parse_args()

image_in = args.image_in
labels_out = args.labels_out

background_image = args.background_image
foreground_image = args.foreground_image
boundary_image = args.boundary_image
raw_image = args.raw_image

f = image_in

# os.environ["KERAS_BACKEND"] =  "tensorflow"
# os.environ["KERAS_BACKEND"] =  "cntk"

# base_dir = os.path.expanduser("/Users/ctr26/Desktop/npl_ftp/")
# base_dir = os.path.expanduser("~/npl_ftp/")
# base_dir = os.path.expanduser("~/mnt/gdrive/data/_cellesce/2D_dapi/data/")

# to_glob = os.path.join(base_dir, "**", "*DAPI*", "projection_*bit.tif")
# files = glob.glob(to_glob, recursive=True)
# shuffle(files)

image_dims = (2048, 2048)

# print(f"Begin in folder {to_glob}")
# print(f"Found {str(len(files))} files")
# print(files)
# folder = "~/unet-nuclei/testimages"
# files = os.listdir(folder)
# files = [os.path.join(folder,f) for f in files]
# files

model = unet_nuclei.unet_initialize(image_dims, automated_shape_adjustment=True)
# %%
# for i, f in enumerate(tqdm(files)):
# try:
    # print(f"Image {str(i)} of {str(len(files))}")
    # basename = os.path.basename(f)
    # base, ext = os.path.splitext(basename)
    # dirname = os.path.dirname(f)
    # filename = os.path.join(dirname, base+"_unet")
im = imageio.imread(image_in)
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
# plt.imshow(im); plt.show()
prediction = unet_nuclei.unet_classify(model, im_padded)

# plt.imsave(filename+".png",prediction)
from skimage import io


io.imsave(background_image,prediction[:, :, 0])
io.imsave(foreground_image,prediction[:, :, 1])
io.imsave(boundary_image,prediction[:, :, 2])
io.imsave(raw_image,im_padded)

im_padded
# io.imsave(labels_out,prediction)

imageio.mimwrite(
    labels_out, np.array([im_padded,
                            prediction[:, :, 0],
                            prediction[:, :, 1],
                            prediction[:, :, 2]]))

# plt.imshow(prediction); plt.show()

# Image.fromarray(prediction).save(filename)
# print(f"Saved image at {filename}")

# except:
#     print(f"Failed on {f}")

# %%
