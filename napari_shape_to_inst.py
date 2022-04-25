# %%
import pandas as pd
import skimage
import matplotlib.pyplot as plt
import numpy as np
from napari import layers
from glob import glob
from pathlib import Path
from tqdm import tqdm
# %%
file = "data/_2019_cellesce/2019-05-20/190520_14.32.57_ISO49_Vemurafenib_0_37uM/Position_2/190520_14.32.57_Step_Size_+0.4_Wavelength_DAPI 452-45_Position_Position_2_ISO49_Vemurafenib_0_37uM/shapes.csv"
glob_files = glob("data/**/shapes.csv",recursive=True)
# %%
for glob_file in tqdm(glob_files):
    # print(folder)
    path = Path(glob_file)
    folder = path.parent.absolute()
    try:
        df = pd.read_csv(glob_file)
    except:
        print(f'Failed on {folder}')
        continue
    image_shape = (2048, 2048)

    # zero = df[df["index"] == 2].set_index("vertex-index")
    # mask = skimage.draw.polygon2mask(image_shape, zero[["axis-0", "axis-1"]])
    # plt.imshow(mask)
    # plt.show()

    # plt.scatter(zero[["axis-0"]],zero[["axis-1"]],label=zero.index)
    # plt.show()

    mask_list = []
    points_list = []
    for group_name, df_group in df.groupby("index"):
        points = df_group.set_index("vertex-index")[["axis-0", "axis-1"]]
        points_list.append(points)
        mask = skimage.draw.polygon2mask(image_shape, points)*int(group_name)
        mask_list.append(mask)
        instance = np.max(mask_list,axis=0)
    skimage.io.imsave(f"{folder}/projection_XY_16_bit_manual_instance.png",instance)
    skimage.io.imsave(f"{folder}/projection_XY_16_bit_manual_instance.tif",instance)

    # %%
    # plt.imshow(instance)
    # plt.show()

    # plt.imshow(mask)
    # plt.show()

    # plt.scatter(points[["axis-0"]],points[["axis-1"]],label=points.index)
    # plt.show()

    # # %%
    # # points=zero[["axis-0", "axis-1"]]
    # layer = layers.Shapes([np.array(points)],shape_type=["polygon"])

    # layer = layers.Shapes(points_list,shape_type=["polygon"])
    # labels = layer.to_labels(labels_shape=image_shape)

    # # sorted(l, key=lambda e: sum((a-b)**2 for a,b in zip(e, [2,5])))
    # plt.imshow(labels)

    # # layers.Shapes().load_from   
    # # %%
    # out_file = "projection_XY_16_bit_manual_instance.tiff"
    # # %%
