import os
import numpy as np
from cellpose import models, io

# Directory containing images
img_dir = 'images/'

# Directory containing masks
mask_dir = 'masks/'

# List to hold images and masks
imgs = []
masks = []

# Read in images and masks
for fname in os.listdir(img_dir):
    if fname.endswith('c=1.png'): # assuming png images
        img = io.imread(os.path.join(img_dir, fname))
        mask = io.imread(os.path.join(mask_dir, fname))
        imgs.append(img)
        masks.append(mask)

# Convert lists to numpy arrays
imgs = np.array(imgs)
masks = np.array(masks)

# Create model
model = models.Cellpose(gpu=True, model_type='cyto')

# Set parameters
diam_mean = 30  # mean diameter of objects in image
nimg = len(imgs)  # number of images to train on
learning_rate = 0.2  # learning rate
batch_size = 8  # batch size
n_epochs = 200  # number of epochs

# Train the model
model.train(imgs, masks, learning_rate, batch_size, n_epochs, channels=[0,0])

# Save the model
model.save_model('cellpose_model')