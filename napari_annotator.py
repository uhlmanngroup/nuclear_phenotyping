# %%
import napari
import glob
from qtpy.QtWidgets import QPushButton, QWidget, QMainWindow, QVBoxLayout
import random
from pathlib import Path

# # Define the main window class
# class FancyGUI(QMainWindow):
#     def __init__(self, napari_viewer):          # include napari_viewer as argument (it has to have this name)
#         super().__init__()
#         self.viewer = napari_viewer
#         # self.UI_FILE = str(Path(__file__).parent / "flood_tool.ui")  # path to .ui file
#         # uic.loadUi(self.UI_FILE, self)           # load QtDesigner .ui file


files = glob.glob("data/**/*DAPI*/projection_XY_16_bit.tif", recursive=True)


# viewer.window.add_dock_widget(QPushButton('Next'), area='left')


# flood_widget = FancyGUI(viewer)                                     # Create instance from our class
# viewer.window.add_dock_widget(flood_widget, area='right')


import napari
from skimage.io import imread
from qtpy.QtWidgets import QMainWindow
from qtpy import uic
from pathlib import Path
from qtpy.QtWidgets import QPushButton


def flood(image, delta):
    new_level = delta * 85
    label_image = image <= new_level
    label_image = label_image.astype(int) * 13  # label 13 is blue in napari
    return (label_image, new_level)


# Define the main window class
class SideBar(QWidget):
    def __init__(self,viewer,files):
        self.files = (file for file in random.sample(files, len(files)))
        self.viewer = viewer
        
        super(SideBar, self).__init__()
        self.layout = QVBoxLayout(self)

        self.main_widget = QWidget(self)

        self.button = QPushButton("Next")
        self.button.clicked.connect(self.next_image)
        self.layout.addWidget(self.button)
        
        self.button_masks = QPushButton("Save Masks")
        self.button_masks.clicked.connect(self.save_mask)
        self.layout.addWidget(self.button_masks)
        
        self.setLayout(self.layout)
        
        # self.current_file = files[0]
        self.next_image()

        # self.main_widget.setLayout(self.main_layout)
        # self.setCentralWidget(self.main_widget)

    def next_image(self):
        # image = imread(next(self.files))
        self.viewer.layers.clear()
        self.current_file = Path(next(self.files))
        self.viewer.open(self.current_file)
        self.viewer.add_shapes(shape_type=['polygon'])
        
    def save_mask(self):
        path = self.current_file.parent.absolute()
        self.viewer.layers[-1].save(f"{path}/shapes")
        print(path)
        # self.viewer.layers.clear()
        
viewer = napari.Viewer()
# napari_image = imread(files[0])  # Reads an image from file
# viewer.add_image(
#     napari_image, name="napari_island"
# )  # Adds the image to the viewer and give the image layer a name

widget = SideBar(viewer,files)  # Create instance from our class
viewer.window.add_dock_widget(
    widget, area="right"
)  # Add our gui instance to napari viewer

# %%

# class FancyGUI(QMainWindow):
#     def __init__(
#         self, napari_viewer
#     ):  # include napari_viewer as argument (it has to have this name)
#         super().__init__()
#         self.viewer = napari_viewer
#         # self.UI_FILE = str(Path(__file__).parent / "flood_tool.ui")  # path to .ui file
#         # uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file

#         self.label_layer = None  # stored label layer variable
#         self.
#         self.pushButton.clicked.connect(self.apply_delta)

#     def apply_delta(self):
#         image = self.viewer.layers[
#             "napari_island"
#         ].data  # We chose to use the layer name to find the correct image layer
#         delta = self.doubleSpinBox.value()
#         label, level = flood(image, delta)
#         if self.label_layer is None:
#             self.label_layer = self.viewer.add_labels(label)
#         else:
#             self.label_layer.data = label
#         self.horizontalSlider.setValue(level)
print("ok")
