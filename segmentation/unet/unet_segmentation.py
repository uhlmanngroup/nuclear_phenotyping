# %%
from unet_nuclei import *
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio
import os
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

# base_dir = os.path.expanduser("/Users/ctr26/Desktop/npl_ftp/")
base_dir = os.path.expanduser("~/npl_ftp/")
base_dir = os.path.expanduser("~/mnt/gdrive/data/_cellesce/2D_dapi/data/")

to_glob = os.path.join(base_dir, "**", "*DAPI*", "projection_*bit.tif")
files = glob.glob(to_glob, recursive=True)
shuffle(files)

image_dims = (2048, 2048)

# print(f"Begin in folder {to_glob}")
# print(f"Found {str(len(files))} files")
# print(files)
# folder = "~/unet-nuclei/testimages"
# files = os.listdir(folder)
# files = [os.path.join(folder,f) for f in files]
# files

model = unet_initialize(image_dims, automated_shape_adjustment=True)
# %%
for i, f in enumerate(tqdm(files)):
    try:
        print(f"Image {str(i)} of {str(len(files))}")
        basename = os.path.basename(f)
        base, ext = os.path.splitext(basename)
        dirname = os.path.dirname(f)
        filename = os.path.join(dirname, base+"_unet")
        im = imageio.imread(f)
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
        prediction = unet_classify(model, im_padded)

        # plt.imsave(filename+".png",prediction)
        imageio.mimwrite(
            filename+".tif", np.array([im_padded,
                                    prediction[:, :, 0],
                                    prediction[:, :, 1],
                                    prediction[:, :, 2]]))
        # plt.imshow(prediction); plt.show()

        # Image.fromarray(prediction).save(filename)
        print(f"Saved image at {filename}")

    except:
        print(f"Failed on {f}")

# %%
