import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import pandas as pd
import shutil
import torch
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.request import urlretrieve

from napari import layers
from napari.utils import progress
from napari.utils.notifications import show_info, show_error, show_warning
from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from skimage.color import rgb2gray
from skimage.io import imsave

from napari_organoid_analyzer import _utils as utils
from napari_organoid_analyzer import session
from napari_organoid_analyzer import settings
from napari_organoid_analyzer._orgacount import OrganoiDL
from napari_organoid_analyzer._utils import (
    compute_image_hash,
    convert_boxes_from_napari_view,
    convert_boxes_to_napari_view,
    get_timelapse_name,
    polygon2mask,
    validate_bboxes,
)
from napari_organoid_analyzer._widgets.annotation import get_annotation_dialogue
from napari_organoid_analyzer._widgets.dialogues import (
    ConfirmSamUpload,
    ConfirmUpload,
    ExportDialog,
    SignalChannelDialog,
    SignalDialog,
)
from napari_organoid_analyzer._training.training_widget import TrainingWidget

warnings.filterwarnings("ignore")

ANNOTATION_TYPES = ['Text', 'Ruler', 'Objects / Boxes', 'Classes', 'Number']

class OrganoidAnalyzerWidget(QWidget):
    '''
    The main widget of the organoid analyzer
    Parameters
    ----------
        napari_viewer: string
            The current napari viewer
        window_sizes: list of ints, default [1024]
            A list with the sizes of the windows on which the model will be run. If more than one window_size is given then the model will run on several window sizes and then 
            combine the results
        downsampling:list of ints, default [2]
            A list with the sizes of the downsampling ratios for each window size. List size must be the same as the window_sizes list
        min_diameter: int, default 30
            The minimum organoid diameter given in um
        confidence: float, default 0.8
            The confidence threhsold - equivalent to box_score_thresh of faster_rcnn
    Attributes
    ----------
        model_name: str
            The name of the model user has selected
        image_layer_names: list of strings
            Will hold the names of all the currently open images in the viewer
        image_layer_name: string
            The image we are currently working on
        shape_layer_names: list of strings
            Will hold the names of all the currently open images in the viewer
        save_layer_name: string
            The name of the shapes layer that has been selected for saving
        cur_shapes_name: string
            The name of the shapes layer that has been selected for visualisation
        cur_shapes_layer: napari.layers.Shapes
            The current shapes layer we are working on - it's name should correspond to cur_shapes_name
        organoiDL: OrganoiDL
            The class in which all the computations are performed for computing and storing the organoids bounding boxes and confidence scores
        num_organoids: int
            The current number of organoids
        original_images: dict
        original_contrast: dict
        label2im: dict
            Stores a mapping between label layer names and image layer names
    '''
    def __init__(self, 
                napari_viewer,
                window_sizes: List = [1024],
                downsampling: List = [2],
                window_overlap: float = 0.5,
                min_diameter: int = 30,
                confidence: float = 0.8):
        super().__init__()

        # assign class variables
        self.viewer = napari_viewer 

        # create cache dir for models if it doesn't exist and add any previously added local
        # models to the model dict
        settings.init()
        utils.add_local_models()
        session.load_cached_settings()
        self.model_id = 2 # yolov3
        self.model_name = list(settings.MODELS.keys())[self.model_id]
        
        # init params 
        self.window_sizes = window_sizes
        self.downsampling = downsampling
        self.window_overlap = window_overlap
        self.min_diameter = min_diameter
        self.confidence = confidence

        self.image_layer_names = []
        self.image_layer_name = None
        self.label_layer_name = None
        self.shape_layer_names = []
        self.cur_shapes_layer = None
        self.num_organoids = 0
        self.original_images = {}
        self.original_contrast = {}
        self.stored_confidences = {}
        self.stored_diameters = {}
        self.label2im = {}
        self.timelapses = {}
        self.cur_timelapse_name = None
        self.timelapse_image_layers = set()
        self.timelapse_segmentations = {}
        self.im2signal = {}


        # Add cache-related attributes
        self.cache_enabled = True  # Default to enabled
        self.image_hashes = {}  # Store image hashes for quick lookup
        self.cache_index_file = os.path.join(str(settings.DETECTIONS_DIR), "cache_index.json")
        self.cache_index = self._load_cache_index()
        self.remember_choice_for_image_import = None  # Variable to store user choice for image import

        # Initialize guided mode to False
        self.guided_mode = False
        self.guidance_layer_name = None
        self.guidance_layers = set()

        # Setup tab widget
        self.tab_widget = QTabWidget()
        self.configuration_tab = QWidget()
        self.detection_data_tab = QWidget()
        self.annotation_tab = QWidget()
        self.timelapse_tab = QWidget()
        self.training_tab = TrainingWidget(parent=self)

        # Setup tabs for the widget
        self.tab_widget.addTab(self.configuration_tab, "Configuration")
        self.tab_widget.addTab(self.detection_data_tab, "Detection data")
        self.tab_widget.addTab(self.annotation_tab, "Add Annotation")
        self.tab_widget.addTab(self.timelapse_tab, "TL and Tracking")
        self.tab_widget.addTab(self.training_tab, "Training")

        # Set up the layout for the configuration tab
        self.configuration_tab.setLayout(QVBoxLayout())
        self.configuration_tab.layout().addWidget(self._setup_input_widget())
        self.configuration_tab.layout().addWidget(self._setup_output_widget())
        self.configuration_tab.layout().addWidget(self._setup_segmentation_widget())

        # Set up the layout for the detection data tab
        self.detection_data_tab.setLayout(QVBoxLayout())
        self.detection_data_tab.layout().addWidget(self._setup_search_tool_widget())
        self.selected_data_group = QGroupBox("Selected data")
        selected_data_vbox = QVBoxLayout()
        self.detection_data_tree = QTreeWidget()
        self.detection_data_tree.setHeaderLabels(["Detections", "Properties"])
        selected_data_vbox.addWidget(self.detection_data_tree)
        self.selected_data_group.setLayout(selected_data_vbox)
        self.detection_data_tab.layout().addWidget(self.selected_data_group)
        export_button = QPushButton("Export Selected")
        export_button.clicked.connect(self._export_detection_data_to_csv)
        self.detection_data_tab.layout().addWidget(export_button)

        # Set up the layout for the add annotation tab
        self.annotation_tab.setLayout(QVBoxLayout())
        self.annotation_tab.layout().addWidget(self._setup_labels_for_annotation_widget())
        self.annotation_tab.layout().addWidget(self._setup_create_annotation_feature_widget())
        self.annotation_tab.layout().addWidget(self._setup_continue_annotation_widget())
        self.annotation_tab.layout().addStretch()

        # Set up the layout for the TL & Tracking tab
        self.timelapse_tab.setLayout(QVBoxLayout())
        self.timelapse_tab.layout().addWidget(self._setup_timelapse_widget())

        # Add the tab widget to the main layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.tab_widget)

        # initialise organoidl instance
        self.organoiDL = OrganoiDL(self.handle_progress)

        # get already opened layers
        image_layer_names = self._get_layer_names()  # Must not be self.image_layer_names, otherwise it will recursively add the same image again and again.
        if len(image_layer_names)>0: 
            self._update_added_image(image_layer_names)
        shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
        if len(shape_layer_names)>0: 
            self._update_added_shapes(shape_layer_names)
        # and watch for newly added images or shapes
        self.viewer.layers.events.inserted.connect(self._added_layer)
        self.viewer.layers.events.removed.connect(self._removed_layer)
        self.viewer.layers.selection.events.changed.connect(self._sel_layer_changed)
        self.viewer.dims.events.current_step.connect(self._on_frame_change)
        for layer in self.viewer.layers:
            layer.events.name.connect(self._on_layer_name_change)
    
        # setup flags used for changing slider and text of min diameter and confidence threshold
        self.diameter_slider_changed = False 
        self.confidence_slider_changed = False
        self.diameter_textbox_changed = False
        self.confidence_textbox_changed = False

    def handle_progress(self, blocknum, blocksize, totalsize):
        """ When the model is being downloaded, this method is called and th progress of the download
        is calculated and displayed on the progress bar. This function was re-implemented from:
        https://www.geeksforgeeks.org/pyqt5-how-to-automate-progress-bar-while-downloading-using-urllib/ """
        read_data = blocknum * blocksize # calculate the progress
        if totalsize > 0:
            download_percentage = read_data * 100 / totalsize
            self.progress_bar.setValue(int(download_percentage))
            QApplication.processEvents()

    def _load_cache_index(self):
        """Load the cache index from disk or create a new one if it doesn't exist"""
        if os.path.exists(self.cache_index_file):
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                show_warning("No detections cache found")
        return {}
    
    def _save_cache_index(self):
        """Save the cache index to disk"""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f)
        except IOError:
            show_error("Failed to save cache index")
    
    def _check_for_cached_results(self, image_hash):
        """Check if the given image hash has cached detection results"""
        if image_hash in self.cache_index:
            cache_file = self.cache_index[image_hash]
            if os.path.exists(cache_file):
                return cache_file
            else:
                show_error(f"Cache file {cache_file} not found although present in cache index")
        return None
    
    def _save_cache_results(self, layer_name):
        if not self.cache_enabled:
            return

        if layer_name not in self.label2im:
            show_error(f"Layer {layer_name} doesn't have associated image layer")
            return
        
        corr_image_name = self.label2im[layer_name]
        
        if corr_image_name not in self.viewer.layers:
            show_error(f"Image layer {self.label2im[layer_name]} not found in viewer")
            return
        
        if layer_name.startswith("TL_Frame"):
            image_hash = self.image_hashes[f"{layer_name.split('_')[1]}_{corr_image_name}"]
        else:
            image_hash = self.image_hashes[self.label2im[layer_name]]

        cache_file = os.path.join(
            str(settings.DETECTIONS_DIR), 
            f"cache_{image_hash}.json"
        )
        
        # Create a dictionary to store the data
        scale = self.viewer.layers[corr_image_name].scale[:2]
        confidence = self.stored_confidences.get(layer_name, self.confidence)
        min_diameter = self.stored_diameters.get(layer_name, self.min_diameter)

        cache_data = {
            'scale': scale.tolist(),
            'confidence': confidence,
            'min_diameter': min_diameter
        }
        cache_data.update(self.organoiDL.storage.get(layer_name, {}))
        
        # Write the data to the cache file
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
                
        self.cache_index[image_hash] = cache_file
        self._save_cache_index()

    
    def _load_cached_results(self, cache_file):
        """Load detection results from cache file"""
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            show_error(f"Failed to load cached results from {cache_file}")
            return None
        
        if "detection_data" in cache_data:
            # Convert all keys (bbox_ids) to integers
            cache_data['detection_data'] = {
                int(k): v for k, v in cache_data['detection_data'].items()
            }
        if "segmentation_data" in cache_data:
            cache_data['segmentation_data'] = {
                int(k): v for k, v in cache_data.get('segmentation_data', {}).items()
            }
        if "annotation_data" in cache_data:
            cache_data['annotation_data'] = {
                int(k): v for k, v in cache_data.get('annotation_data', {}).items()
            }
        return cache_data
        
    
    def _create_shapes_from_cache(self, image_layer_name, cache_data, labels_layer_name=None):
        """Create a shapes layer from cached detection data"""
        if self.organoiDL.img_scale[0] == 0:
            self.organoiDL.set_scale(self.viewer.layers[image_layer_name].scale[:2])

        scale = cache_data.pop('scale', self.viewer.layers[image_layer_name].scale[:2])
        confidence = cache_data.pop('confidence', self.confidence)
        min_diameter = cache_data.pop('min_diameter', self.min_diameter)


        if scale[0] != self.viewer.layers[image_layer_name].scale[0] or scale[1] != self.viewer.layers[image_layer_name].scale[1]:
            show_warning("Scale mismatch between cached data and current image layer")

        if len(cache_data.get('detection_data', {})) == 0:
            show_error("No detections found in cache")
            return False
            
        # Create a new shapes layer
        if labels_layer_name is None:
            labels_layer_name = f'{image_layer_name}-Labels-Cache-{datetime.strftime(datetime.now(), "%H_%M_%S")}'
        
        self.organoiDL.storage[labels_layer_name] = cache_data
        self._update_detections(
            labels_layer_name=labels_layer_name, 
            confidence=confidence, 
            min_diameter=min_diameter, 
            image_layer_name=image_layer_name
        )
        return True

    def _sel_layer_changed(self, event):
        """ Is called whenever the user selects a different layer to work on. """
        cur_layer_list = list(self.viewer.layers.selection)
        if len(cur_layer_list)==0: return
        cur_seg_selected = cur_layer_list[-1]
        if self.cur_shapes_layer and cur_seg_selected.name == self.cur_shapes_layer.name: return
        # switch to values of other shapes layer if clicked
        if type(cur_seg_selected)==layers.Shapes and not cur_seg_selected.name in self.guidance_layers:
            if self.cur_shapes_layer is not None:
                self.stored_confidences[self.cur_shapes_layer.name] = self.confidence
                self.stored_diameters[self.cur_shapes_layer.name] = self.min_diameter
            self.cur_shapes_layer = cur_seg_selected
            # update min diameter text and slider with previous value of that layer
            self.min_diameter = self.stored_diameters[self.cur_shapes_layer.name]
            self.min_diameter_textbox.setText(str(self.min_diameter))
            self.min_diameter_slider.setValue(self.min_diameter)
            # update confidence text and slider with previous value of that layer
            self.confidence = self.stored_confidences[self.cur_shapes_layer.name]
            self.confidence_textbox.setText(str(self.confidence))
            self.confidence_slider.setValue(int(self.confidence*100))
            self._update_num_organoids(len(self.cur_shapes_layer.data))
            # update label and checkbox for current shapes layer
            self._update_cur_shapes_layer_label_and_checkbox()

    def _update_cur_shapes_layer_label_and_checkbox(self):
        """Update the label and checkbox for the current shapes layer name and timelapse option."""
        self.cur_shapes_layer_label.setText(f"Current shapes layer: {self.cur_shapes_layer.name if self.cur_shapes_layer else 'None'}")
        self.search_cur_shapes_layer_label.setText(f"Current shapes layer: {self.cur_shapes_layer.name if self.cur_shapes_layer else 'None'}")
        if self.cur_shapes_layer and self.cur_shapes_layer.name.startswith("TL_Frame"):
            self.apply_to_timelapse_checkbox.setVisible(True)
        else:
            self.apply_to_timelapse_checkbox.setVisible(False)
        
        segmentation_selection_items = [self.segmentation_image_layer_selection.itemText(i) 
                                        for i in range(self.segmentation_image_layer_selection.count())]
        if self.cur_shapes_layer:
            if self.cur_shapes_layer.name in segmentation_selection_items:
                self.segmentation_image_layer_selection.setCurrentText(self.cur_shapes_layer.name)
            else:
                show_error(f"Current shapes layer '{self.cur_shapes_layer.name}' not found in segmentation layer selection")
        else:
            self.segmentation_image_layer_selection.setCurrentText('')

    def _added_layer(self, event):
        # get names of added layers, image and shapes
        new_image_layer_names = self._get_layer_names()
        new_shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
        new_image_layer_names = [name for name in new_image_layer_names if name not in self.image_layer_names]
        new_shape_layer_names = [name for name in new_shape_layer_names if name not in self.shape_layer_names]
        if len(new_image_layer_names)>0 : 
            self._update_added_image(new_image_layer_names)
        if len(new_shape_layer_names)>0:
            self._update_added_shapes(new_shape_layer_names)
            self.shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)

        for layer in self.viewer.layers:
            layer.events.name.connect(self._on_layer_name_change)
            if type(layer) == layers.Shapes:
                layer.events.highlight.connect(self._on_shape_selected)

    def compute_and_check_image_hash(self, image_data, image_name, shapes_name=None):
        image_hash = compute_image_hash(image_data)

        if shapes_name and shapes_name.startswith("TL_Frame"):
            save_name = f"{shapes_name.split('_')[1]}_{image_name}"
            self.image_hashes[save_name] = image_hash
        else:
            self.image_hashes[image_name] = image_hash
        
        # If the user has already chosen to remember their choice, use it
        if self.remember_choice_for_image_import is not None:
            if self.remember_choice_for_image_import:
                cache_file = self._check_for_cached_results(image_hash)
                if cache_file:
                    cache_data = self._load_cached_results(cache_file)
                    if cache_data:
                        self._create_shapes_from_cache(image_name, cache_data, shapes_name)
                        return True
            return False
        cache_file = self._check_for_cached_results(image_hash)
        if cache_file:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle('Cached Results Available')
            if shapes_name and shapes_name.startswith("TL_Frame"):
                msg_box.setText(f"Found cached detection results for timelapse {image_name} ({shapes_name.split('_')[1]}). Load them?")
            else:
                msg_box.setText(f'Found cached detection results for image {image_name}. Load them?')
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            
            # Add a checkbox to remember the user's choice
            remember_checkbox = QCheckBox("Remember my choice for this image import")
            layout = QVBoxLayout()
            layout.addWidget(remember_checkbox)
            msg_box.setCheckBox(remember_checkbox)

            reply = msg_box.exec()
            remember_choice = remember_checkbox.isChecked()

            # Save the user's choice if they selected "Remember"
            if remember_choice:
                self.remember_choice_for_image_import = (reply == QMessageBox.Yes)

            if reply == QMessageBox.Yes:
                cache_data = self._load_cached_results(cache_file)
                if cache_data:
                    self._create_shapes_from_cache(image_name, cache_data, shapes_name)
                    return True
        return False

    def _removed_layer(self, event):
        """ Is called whenever a layer has been deleted (by the user) and removes the layer from GUI and backend. """
        new_image_layer_names = self._get_layer_names()
        new_shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
        removed_image_layer_names = [name for name in self.image_layer_names if name not in new_image_layer_names]
        removed_shape_layer_names = [name for name in self.shape_layer_names if name not in new_shape_layer_names]
        if len(removed_image_layer_names)>0:
            self._update_removed_image(removed_image_layer_names)
            self.image_layer_names = self._get_layer_names()
        if len(removed_shape_layer_names)>0:
            self._update_remove_shapes(removed_shape_layer_names)
            self.shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)

    def _preprocess(self, layer_name, img):
        """ Preprocess the current image in the viewer to improve visualisation for the user """
        img = utils.apply_normalization(img)
        self.viewer.layers[layer_name].data = img
        self.viewer.layers[layer_name].contrast_limits = (0,255)

    def _update_num_organoids(self, len_bboxes):
        """ Updates the number of organoids displayed in the viewer """
        self.num_organoids = len_bboxes
        new_text = 'Number of organoids: '+str(self.num_organoids)
        self.organoid_number_label.setText(new_text)

    def _update_detections(self, labels_layer_name, confidence=None, min_diameter=None, image_layer_name=None):
        """ 
        Adds the shapes layer to the viewer or updates it if already there.
        Required layer info to already be in self.organoiDL.storage dict.
        """
        if not labels_layer_name in self.label2im:
            if image_layer_name is None:
                show_error(f"Image layer name not provided for {labels_layer_name}.")
                return
            self.label2im[labels_layer_name] = image_layer_name
        if confidence is None:
            confidence = self.confidence
        if min_diameter is None:
            min_diameter = self.min_diameter
        bboxes, properties = self.organoiDL.apply_params(
            labels_layer_name,
            confidence,
            min_diameter,
        )
        self._update_num_organoids(len(bboxes))
        if labels_layer_name in self.shape_layer_names:
            self.viewer.layers[labels_layer_name].data = bboxes
            self.viewer.layers[labels_layer_name].properties = properties
            self.viewer.layers[labels_layer_name].edge_width = 12
            self.viewer.layers[labels_layer_name].refresh()
            self.viewer.layers[labels_layer_name].refresh_text()
            self.cur_shapes_layer = self.viewer.layers[labels_layer_name]
        else:
            text_params = {'string': 'ID: {bbox_id}\nConf.: {score:.2f}',
                            'size': 12,
                            'anchor': 'upper_left',
                            'color': settings.TEXT_COLOR}
            # if no organoids were found just make an empty shapes layer
            if self.num_organoids==0: 
                self.cur_shapes_layer = self.viewer.add_shapes(name=labels_layer_name,
                                                               properties=properties,
                                                               text=text_params,
                                                               edge_color=settings.COLOR_DEFAULT,
                                                               face_color='transparent',
                                                               edge_width=12,
                                                               scale=self.viewer.layers[image_layer_name].scale[:2],)
            # otherwise make the layer and add the boxes
            else:
                self.cur_shapes_layer = self.viewer.add_shapes(bboxes, 
                                                               name=labels_layer_name,
                                                               scale=self.viewer.layers[image_layer_name].scale[:2],
                                                               face_color='transparent',  
                                                               properties = properties,
                                                               text = text_params,
                                                               edge_color=settings.COLOR_DEFAULT,
                                                               shape_type='rectangle',
                                                               edge_width=12)
            if labels_layer_name.startswith("TL_Frame"):
                timelapse_name = get_timelapse_name(labels_layer_name)
                if not timelapse_name in self.timelapses:
                    self.timelapses[timelapse_name] = set()
                    self.timelapse_selection.addItem(timelapse_name)
                    self.timelapse_selection.setCurrentText(timelapse_name)
                    self.cur_timelapse_name = timelapse_name
                self.timelapses[timelapse_name].add(labels_layer_name)

        if not 'napari-organoid-analyzer:_rerun' in self.cur_shapes_layer.metadata:
            with utils.set_dict_key(self.cur_shapes_layer.metadata, 'napari-organoid-analyzer:_rerun', True):
                self.stored_confidences[labels_layer_name] = confidence
                self.stored_diameters[labels_layer_name] = min_diameter
                self.confidence_slider.setValue(int(confidence * 100))
                self.min_diameter_slider.setValue(int(min_diameter))
                            
        # set current_edge_width so edge width is the same when users annotate - doesnt' fix new preds being added!
        self.viewer.layers[labels_layer_name].current_edge_width = 12
        self.viewer.layers[labels_layer_name].mode = 'select'
        self._save_cache_results(labels_layer_name)
        self._update_cur_shapes_layer_label_and_checkbox()
        self.cur_shapes_layer.events.data.connect(self.shapes_event_handler)

    def _check_sam(self):
        # check if SAM model exists locally and if not ask user if it's ok to download
        if not utils.return_is_file(settings.MODELS_DIR, settings.SAM_MODEL["filename"]): 
            confirm_window = ConfirmSamUpload(self)
            confirm_window.exec()
            # if user clicks cancel return doing nothing 
            if confirm_window.result() != QDialog.Accepted: return
            # otherwise download model and display progress in progress bar
            else: 
                self.progress_box.show()
                save_loc = os.path.join(str(settings.MODELS_DIR),  settings.SAM_MODEL["filename"])
                urlretrieve(settings.SAM_MODEL["url"], save_loc, self.handle_progress)
                self.progress_box.hide()

    def _on_run_click(self):
        """ Is called whenever Run Organoid Counter button is clicked """
        # check if an image has been loaded
        if not self.image_layer_name: 
            show_info('Please load an image first and try again!')
            return
        
        if not self.image_layer_name in self.viewer.layers:
            show_error(f"Image layer {self.image_layer_name} not found in the viewer.")
            return
        
        self._check_sam()

        # check if model exists locally and if not ask user if it's ok to download
        if not utils.return_is_file(settings.MODELS_DIR, settings.MODELS[self.model_name]["filename"]): 
            confirm_window = ConfirmUpload(self, self.model_name)
            confirm_window.exec()
            # if user clicks cancel return doing nothing 
            if confirm_window.result() != QDialog.Accepted: return
            # otherwise download model and display progress in progress bar
            else: 
                self.progress_box.show()
                self.organoiDL.download_model(self.model_name)
                self.progress_box.hide()
        
        # load model checkpoint
        self.organoiDL.set_model(self.model_name)
        if self.organoiDL.img_scale[0] == 0: 
            self.organoiDL.set_scale(self.viewer.layers[self.image_layer_name].scale[:2])
        
        # make sure the number of windows and downsamplings are the same
        if len(self.window_sizes) != len(self.downsampling): 
            show_info('Keep number of window sizes and downsampling the same and try again!')
            return
        
        if not self.guidance_layer_name is None and (not self.guidance_layer_name or self.guidance_layer_name not in self.viewer.layers):
            show_error("Guidance layer not found in the viewer. Please select a valid guidance layer.")
            return
        
        # get the current image 
        img_data = self.viewer.layers[self.image_layer_name].data

        if img_data.ndim == 3 or img_data.ndim == 2:
            if not self.guidance_layer_name is None and not validate_bboxes(self.viewer.layers[self.guidance_layer_name].data, img_data.shape[:2]):
                show_error(f"Bboxes from guidance layer {self.guidance_layer_name} cannot be applied to image {self.image_layer_name} with shape {img_data.shape[:2]}")
                return
            labels_layer_name = f'{self.image_layer_name}-Labels-{self.model_name}-{datetime.strftime(datetime.now(), "%H_%M_%S")}'
            self.viewer.window._status_bar._toggle_activity_dock(True)
            self._detect_organoids(img_data, labels_layer_name, self.image_layer_name)
        elif img_data.ndim == 4:
            if not self.guidance_layer_name is None and not validate_bboxes(self.viewer.layers[self.guidance_layer_name].data, img_data.shape[1:3]):
                show_error(f"Bboxes from guidance layer {self.guidance_layer_name} cannot be applied to image {self.image_layer_name} with shape {img_data.shape[:2]}")
                return
            timelapse_name = f'{self.image_layer_name}-Labels-{self.model_name}-{datetime.strftime(datetime.now(), "%H_%M_%S")}'
            self.viewer.window._status_bar._toggle_activity_dock(True)
            for i in progress(range(img_data.shape[0])):
                labels_layer_name = f'TL_Frame{i}_{timelapse_name}'
                self._detect_organoids(img_data[i], labels_layer_name, self.image_layer_name)
        else:
            show_error(f"Wrong format for image with shapes {img_data.ndim}")
            
        self.viewer.window._status_bar._toggle_activity_dock(False)

    def _detect_organoids(self, img_data, labels_layer_name, img_layer_name):
        """
        Detect organoids from the image (or timelapse frame) and create a shapes layer
        """

        loaded_cached_data = self.compute_and_check_image_hash(img_data, img_layer_name)
        if loaded_cached_data:
            return

        if img_data.ndim == 3:
            if img_data.shape[2] == 4:
                img_data = img_data[:, :, :3]
            img_data = rgb2gray(img_data)
            img_data = (img_data * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8

        if labels_layer_name in self.shape_layer_names:
            show_info('Found existing labels layer. Please remove or rename it and try again!')
            return 
        
        crops = convert_boxes_from_napari_view(self.viewer.layers[self.guidance_layer_name].data)if not self.guidance_layer_name is None else [[0, 0, img_data.shape[0], img_data.shape[1]]]

        # run inference
        self.organoiDL.run(img_data, 
                           labels_layer_name,
                           self.window_sizes,
                           self.downsampling,
                           self.window_overlap,
                           crops)

        self._update_detections(labels_layer_name, image_layer_name=img_layer_name)

    def _on_run_segmentation(self):
        """
        Is called whether run_segmentation button is clicked
        """
        if not self.label_layer_name:
            show_error("No label layer selected. Please select a label layer and try again.")
            return
        
        if not self.label_layer_name in self.viewer.layers:
            show_error(f"Layer '{self.label_layer_name}' not found in the viewer.")
            return
        
        image_name = self.label2im[self.label_layer_name]
        
        if not image_name in self.viewer.layers:
            show_error(f"Image layer '{image_name}' not found in the viewer. Please upload the image again")
            return
        
        image_data = self.viewer.layers[image_name].data

        merged_signal_data = {}
        if image_name in self.im2signal:
            for signal_name, signal_layer_name in self.im2signal[image_name].items():
                if not signal_layer_name in self.viewer.layers:
                    show_warning(f"Signal layer {signal_layer_name} not found in viewer")
                    continue
                signal_data = self.viewer.layers[signal_layer_name].data
                image_shape = image_data.shape
                signal_shape = signal_data.shape
                if not (
                    (len(signal_shape) == len(image_shape) - 1 and signal_shape == image_shape[:-1]) or 
                    (len(signal_shape) == len(image_shape) and signal_shape[:-1] == image_shape[:-1]) or
                    (len(signal_shape) == 3 and len(image_shape) == 2 and signal_shape[:-1] == image_shape)
                ):
                    show_error(f"Signal dimensions of {signal_shape} do not correspond to image dimensions {image_shape} fo signal {signal_layer_name}. Skipping...")
                    continue
                if len(signal_shape) >= len(image_shape) and len(signal_shape) >= 3:
                    channel_dialog = SignalChannelDialog(self, signal_shape[-1], signal_layer_name)
                    if channel_dialog.exec() != QDialog.Accepted:
                        show_warning(f"No channel selected for signal {signal_layer_name} Skipping...")
                        continue
                    idx = channel_dialog.get_channel_idx()
                    signal_data = np.take(signal_data, idx, -1).squeeze()
                signal_field = np.stack([signal_data, signal_data, signal_data], axis=-1)
                merged_signal_data.update({signal_name: signal_field})
    
        self._check_sam()
        if self.organoiDL.sam_predictor is None:
            self.organoiDL.init_sam_predictor()
        
        self.viewer.window._status_bar._toggle_activity_dock(True)
        
        if self.run_for_timelapse_checkbox.isVisible() and self.run_for_timelapse_checkbox.isChecked():
            timelapse_name = get_timelapse_name(self.label_layer_name)

            if timelapse_name not in self.timelapses:
                show_error(f"Timelapse '{timelapse_name}' not found.")
                self.viewer.window._status_bar._toggle_activity_dock(False)
                return
            
            assert image_name == timelapse_name.split('-Labels')[0]
            
            if not image_name in self.timelapse_image_layers or self.viewer.layers[image_name].data.ndim != 4:
                show_error(f"Image layer '{image_name}' is not a timelapse. Please upload a valid timelapse image.")
                self.viewer.window._status_bar._toggle_activity_dock(False)
                return
            
            mask_vis_shape = list(image_data.shape)
            mask_vis_shape[-1] = 3
            total_frames = image_data.shape[0]      
            final_image = np.zeros(mask_vis_shape, dtype=np.uint8)
            final_signal_seg = {signal_name: np.zeros(mask_vis_shape[:-1]) for signal_name, _ in merged_signal_data.items()}

            for i in progress(range(total_frames)):
                frame_layer_name = f'TL_Frame{i}_{timelapse_name}'
                if not frame_layer_name in self.timelapses[timelapse_name] or not frame_layer_name in self.viewer.layers:
                    show_warning(f"No detection data for frame#{i}. Skipping.")
                    continue
                labels_layer = self.viewer.layers[frame_layer_name]
                bboxes = convert_boxes_from_napari_view(labels_layer.data)
                frame = image_data[i]
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                frame_signal = {signal_name: signal_field[i] for signal_name, signal_field in merged_signal_data.items()}
                # run_segmentation now returns collated masks directly
                collated_mask, collated_signal_masks = self.organoiDL.run_segmentation(frame, frame_layer_name, bboxes, frame_signal)
                final_image[i] = collated_mask
                for signal_name, signal_seg in collated_signal_masks.items():
                    final_signal_seg[signal_name][i] = signal_seg
                self._update_detections(frame_layer_name, image_layer_name=image_name)


            segmentation_layer_name = f"Segmentation-{timelapse_name}"
            self.viewer.add_image(final_image, name=segmentation_layer_name, blending='additive')
            self.timelapse_segmentations[timelapse_name] = segmentation_layer_name

            for signal_name, signal_seg in final_signal_seg.items():
                signal_seg_layer_name = f"Segmentation-{signal_name}-{timelapse_name}"
                self.viewer.add_image(signal_seg, name=signal_seg_layer_name, blending='additive', colormap="red")

        else:

            labels_layer = self.viewer.layers[self.label_layer_name]
            bboxes = convert_boxes_from_napari_view(labels_layer.data)

            if self.label_layer_name.startswith("TL_Frame"):
                frame_idx = int(self.label_layer_name.split('_')[1][5:])
                image_data = image_data[frame_idx]
                merged_signal_data = {signal_name: signal_field[frame_idx] for signal_name, signal_field in merged_signal_data.items()}
            if image_data.ndim == 2:
                image_data = np.stack([image_data, image_data, image_data], axis=-1)
            elif image_data.ndim == 3 and image_data.shape[2] == 4:
                image_data = image_data[:, :, :3]
    
            segmentation_layer_name = f"Segmentation-{self.label_layer_name}-{datetime.strftime(datetime.now(), '%H_%M_%S')}"
            # run_segmentation now returns collated masks directly
            collated_mask, collated_signal_masks = self.organoiDL.run_segmentation(image_data, self.label_layer_name, bboxes, merged_signal_data)
    
            self.viewer.add_image(collated_mask, name=segmentation_layer_name, blending='additive')
            for signal_name, collated_signal_mask in collated_signal_masks.items():
                signal_seg_layer_name = f"Segmentation-{signal_name}-{self.label_layer_name}-{datetime.strftime(datetime.now(), '%H_%M_%S')}"
                self.viewer.add_image(collated_signal_mask, name=signal_seg_layer_name, blending='additive', colormap="red")
            self._update_detections(self.label_layer_name, )
    
        self._update_detection_data_tab()
        self.viewer.window._status_bar._toggle_activity_dock(False)
        show_info("Segmentation completed and added to the viewer.")

    def _on_export_layer_click(self):
        """
        Is called whenever the export button is clicked.
        Exports data from the selected label layer or from timelapse associated with the label layer.
        """
        if not self.label_layer_name:
            show_error("No label layer selected. Please select a label layer and try again.")
            return
        
        if not self.label_layer_name in self.viewer.layers:
            show_error(f"Layer '{self.label_layer_name}' not found in the viewer.")
            return
        
        if self.run_for_timelapse_checkbox.isVisible() and self.run_for_timelapse_checkbox.isChecked():
            if not self.label_layer_name.startswith("TL_Frame"):
                raise RuntimeError("Timelapse export is selected, but the current layer is not a timelapse frame.")
            timelapse_export = True
            timelapse_name = get_timelapse_name(self.label_layer_name)
            if timelapse_name not in self.timelapses:
                raise RuntimeError(f"Timelapse '{timelapse_name}' not found.")
            shapes_name = next(iter(self.timelapses[timelapse_name]))
        else:
            timelapse_export = False
            shapes_name = self.label_layer_name
        
        self.export_data(shapes_name, timelapse_export)

    def _on_export_timelapse_click(self):
        """
        Is called whenever the export timelapse button is clicked.
        Exports data from all timelapse frames.
        """
        if not self.cur_timelapse_name in self.timelapses:
            raise RuntimeError(f"Timelapse '{self.cur_timelapse_name}' not found.")
        shapes_name = next(iter(self.timelapses[self.cur_timelapse_name]))
        
        self.export_data(shapes_name, timelapse_export=True)

    def export_data(self, shapes_name, timelapse_export: bool = False):
        """
        Export data from the selected label layer or from timelapse associated with the label layer.
        """
        
        label_layer = self.viewer.layers[shapes_name] if shapes_name in self.viewer.layers else None
        if label_layer is None:
            show_error(f"Layer '{shapes_name}' not found in the viewer.")
            return
        layer_name = label_layer.name
        
        lengths = [len(v) for v in label_layer.properties.values()]
        if len(set(lengths)) != 1:
            raise RuntimeError("Mismatch in number of masks and labels. Please rerun the segmentation on selected layer.")
        
        if timelapse_export:
            timelapse_name = get_timelapse_name(label_layer.name)
            if timelapse_name not in self.timelapses:
                raise RuntimeError(f"Timelapse '{timelapse_name}' not found.")
            
            timelapse_layers = self.timelapses[timelapse_name]
            available_features = set()
            for layer_name in timelapse_layers:
                if layer_name not in self.viewer.layers:
                    show_warning(f"Layer {layer_name} not found in viewer. Skipping...")
                    continue
                if hasattr(self.viewer.layers[layer_name], 'properties') and len(self.viewer.layers[layer_name].properties) > 0:
                    available_features.update(self.viewer.layers[layer_name].properties.keys())
            masks_available = np.any([len(self.organoiDL.storage.get(layer_name, {}).get("segmentation_data", {})) > 0 for layer_name in timelapse_layers])
            ids_with_masks = set()
            for layer_name in timelapse_layers:
                if "segmentation_data" in self.organoiDL.storage.get(layer_name, {}):
                    ids_with_masks.update(self.organoiDL.storage[layer_name]['segmentation_data'].keys())
            ids_with_masks = list(ids_with_masks)
        else:
            available_features = []
            if hasattr(label_layer, 'properties') and label_layer.properties:
                available_features = [k for k in label_layer.properties.keys()]
            masks_available = len(self.organoiDL.storage.get(layer_name, {}).get("segmentation_data", {}))
            ids_with_masks = self.organoiDL.storage[self.label_layer_name]['segmentation_data'].keys() if masks_available else []
        # Open the export dialog
        export_dialog = ExportDialog(self, available_features, masks_available, ids_with_masks)
        if export_dialog.exec_() != QDialog.Accepted:
            show_warning("Export canceled.")
            return
        
        export_path = export_dialog.get_export_path()
        if not export_path:
            show_error("No export folder selected.")
            return
        export_path = Path(export_path)
        
        export_options = export_dialog.get_export_options()
        selected_features = export_dialog.get_selected_features()
        
        exported_items = []
        
        # Process export based on selected options
        if export_options['layer_data']:
            self._export_layer_data(label_layer, export_path, timelapse_export)
            exported_items.append("layer data")
        
        if export_options['instance_masks']:
            self._export_instance_masks(label_layer, export_path, timelapse_export, export_options['selected_ids'])
            exported_items.append("instance masks")

        if export_options['collated_mask']:
            self._export_collated_masks(label_layer, export_path, timelapse_export)
            exported_items.append("collated mask")
        
        if export_options['features']:
            self._export_features(label_layer, export_path, selected_features, timelapse_export)
            exported_items.append("features")
        
        if exported_items:
            show_info(f"Export completed successfully to {str(export_path)}\nExported: {', '.join(exported_items)}")
        else:
            show_warning("No items were selected for export.")

    def _export_layer_data(self, label_layer, export_path: Path, timelapse_export):
        """Export layer data to JSON file"""
        if timelapse_export:
            
            if not label_layer.name.startswith("TL_Frame"):
                raise RuntimeError("Internal error: Timelapse checkbox is checked but current layer is not a timelapse frame.")
            
            timelapse_name = get_timelapse_name(label_layer.name)

            if timelapse_name not in self.timelapses:
                raise RuntimeError(f"Timelapse '{timelapse_name}' not found.")
            
            timelapse_layers = self.timelapses[timelapse_name]
            data_json = {}
            for label_layer_name in timelapse_layers:
                if not label_layer_name.startswith("TL_Frame"):
                    raise RuntimeError(f"Layer {label_layer_name} is in timelapse but not a timelapse frame")
                if not label_layer_name in self.organoiDL.storage:
                    raise RuntimeError(f"No storage data for layer {label_layer_name}. ")
                frame_idx = label_layer_name.split('_')[1][5:]
                frame_data = self.organoiDL.storage[label_layer_name]
                confidence = self.stored_confidences.get(label_layer_name, self.confidence)
                min_diameter = self.stored_diameters.get(label_layer_name, self.min_diameter)
                frame_data.update({
                    "confidence": confidence,
                    "min_diameter": min_diameter
                })
                data_json.update({frame_idx: frame_data})
            data_json["type"] = "timelapse"            
        else:
        
            if not label_layer.name in self.organoiDL.storage:
                raise RuntimeError(f"No storage data for layer {label_layer.name}. ")
            data_json = self.organoiDL.storage[label_layer.name]
            confidence = self.stored_confidences.get(label_layer.name, self.confidence)
            min_diameter = self.stored_diameters.get(label_layer.name, self.min_diameter)
            data_json.update({
                "confidence": confidence,
                "min_diameter": min_diameter
            })
            data_json["type"] = "layer"
            
        # Write bbox coordinates to json
        json_file_path = export_path / f"{self.label_layer_name}_layer_data.json"
        utils.write_to_json(json_file_path, data_json)

    def _export_instance_masks(self, label_layer, export_path: Path, timelapse_export, selected_ids=[]):
        """Export instance masks as binary masks from storage polygons"""

        if timelapse_export:
            if not label_layer.name.startswith("TL_Frame"):
                raise RuntimeError("Internal error: Timelapse checkbox is checked but current layer is not a timelapse frame.")
            
            timelapse_name = get_timelapse_name(label_layer.name)

            if timelapse_name not in self.timelapses:
                raise RuntimeError(f"Timelapse '{timelapse_name}' not found.")
            
            timelapse_layers = self.timelapses[timelapse_name]
            export_folder = export_path / f"instance_masks_{timelapse_name}"
            export_folder.mkdir(exist_ok=True)
            
            for label_layer_name in timelapse_layers:
                if not label_layer_name in self.viewer.layers:
                    raise RuntimeError(f"Label layer {label_layer_name} not found in viewer")
                if not label_layer_name.startswith("TL_Frame"):
                    raise RuntimeError(f"Layer {label_layer_name} is in timelapse but not a timelapse frame")
                if not label_layer_name in self.organoiDL.storage:
                    show_warning(f"No storage data found for layer {label_layer_name}. Skipping...")
                    continue
                
                storage_data = self.organoiDL.storage[label_layer_name]
                if 'segmentation_data' not in storage_data:
                    show_warning(f"No segmentation data found for layer {label_layer_name}. Skipping...")
                    continue
                
                frame_idx = int(label_layer_name.split('_')[1][5:])
                image_shape = storage_data['image_size']
                mask_dict = {}
                
                for obj_id in selected_ids:
                    if obj_id not in storage_data['segmentation_data']:
                        continue  # Skip IDs that don't exist in this frame
                    
                    obj_masks = {}
                    segmentation_obj_data = storage_data['segmentation_data'][obj_id]
                    
                    for mask_key, polygon_data in segmentation_obj_data.items():
                        if polygon_data:
                            polygon = json.loads(polygon_data)
                            if polygon:  # Check if polygon is not empty
                                binary_mask = polygon2mask(polygon, image_shape)
                                obj_masks[mask_key] = binary_mask
                    
                    if obj_masks:  # Only add to mask_dict if we have masks
                        mask_dict[obj_id] = obj_masks
                
                if mask_dict:
                    file_path = export_folder / f"Frame_{frame_idx}"
                    np.save(file_path, mask_dict)
        else:
            if self.label_layer_name not in self.organoiDL.storage:
                raise RuntimeError(f"No storage data found for layer {self.label_layer_name}.")
            
            storage_data = self.organoiDL.storage[self.label_layer_name]
            if 'segmentation_data' not in storage_data:
                show_warning("No segmentation data found. Skipping mask export.")
                return
            
            image_shape = storage_data['image_size']
            mask_dict = {}
            
            for obj_id in selected_ids:
                if obj_id not in storage_data['segmentation_data']:
                    continue  # Skip IDs that don't exist
                
                obj_masks = {}
                segmentation_obj_data = storage_data['segmentation_data'][obj_id]
                
                # Export all masks (regular mask and signal masks) for this object
                for mask_key, polygon_data in segmentation_obj_data.items():
                    if polygon_data:
                        polygon = json.loads(polygon_data)
                        if polygon:  # Check if polygon is not empty
                            binary_mask = polygon2mask(polygon, image_shape)
                            obj_masks[mask_key] = binary_mask
                
                if obj_masks:  # Only add to mask_dict if we have masks
                    mask_dict[obj_id] = obj_masks
            
            if not mask_dict:
                show_warning("No masks found for selected IDs. Skipping mask export.")
                return
            
            instance_mask_file_path = export_path / f"{self.label_layer_name}_instance_masks.npy"
            np.save(instance_mask_file_path, mask_dict)

    def _export_collated_masks(self, label_layer, export_path: Path, timelapse_export):
        """Export collated masks to NPY"""

        if timelapse_export:
            if not label_layer.name.startswith("TL_Frame"):
                raise RuntimeError("Internal error: Timelapse checkbox is checked but current layer is not a timelapse frame.")
            
            timelapse_name = get_timelapse_name(label_layer.name)

            if timelapse_name not in self.timelapses:
                raise RuntimeError(f"Timelapse '{timelapse_name}' not found.")
            
            timelapse_layers = self.timelapses[timelapse_name]
            export_folder = export_path / f"collated_masks_{timelapse_name}"
            export_folder.mkdir(exist_ok=True)
            
            for label_layer_name in timelapse_layers:
                if not label_layer_name in self.viewer.layers:
                    raise RuntimeError(f"Label layer {label_layer_name} not found in viewer")
                if not label_layer_name.startswith("TL_Frame"):
                    raise RuntimeError(f"Layer {label_layer_name} is in timelapse but not a timelapse frame")
                if not label_layer_name in self.organoiDL.storage:
                    show_warning(f"No storage data found for layer {label_layer_name}. Skipping...")
                    continue
                
                storage_data = self.organoiDL.storage[label_layer_name]
                if 'segmentation_data' not in storage_data:
                    show_warning(f"No segmentation data found for layer {label_layer_name}. Skipping...")
                    continue
                
                frame_idx = int(label_layer_name.split('_')[1][5:])
                image_shape = storage_data['image_size']
                
                # Find all unique mask types across all objects
                mask_types = set()
                for obj_id, seg_data in storage_data['segmentation_data'].items():
                    mask_types.update(seg_data.keys())
                
                # Create a collated mask for each mask type
                collated_masks = {}
                for mask_type in mask_types:

                    collated_mask = np.zeros(image_shape, dtype=np.uint8)
                    
                    # Add all instances of this mask type to the collated mask
                    for obj_id, seg_data in storage_data['segmentation_data'].items():
                        if mask_type in seg_data and seg_data[mask_type]:
                            polygon = json.loads(seg_data[mask_type])
                            if polygon:
                                binary_mask = polygon2mask(polygon, image_shape)
                                collated_mask[binary_mask > 0] = 1
                    
                    if np.any(collated_mask):
                        collated_masks[mask_type] = collated_mask
                
                if collated_masks:
                    file_path = export_folder / f"Frame_{frame_idx}"
                    np.save(file_path, collated_masks)
        else:
            if self.label_layer_name not in self.organoiDL.storage:
                raise RuntimeError(f"No storage data found for layer {self.label_layer_name}.")
            
            storage_data = self.organoiDL.storage[self.label_layer_name]
            if 'segmentation_data' not in storage_data:
                show_warning("No segmentation data found. Skipping mask export.")
                return
            
            image_shape = storage_data['image_size']
            
            # Find all unique mask types across all objects
            mask_types = set()
            for obj_id, seg_data in storage_data['segmentation_data'].items():
                mask_types.update(seg_data.keys())
            
            # Create a collated mask for each mask type
            collated_masks = {}
            for mask_type in mask_types:

                collated_mask = np.zeros(image_shape, dtype=np.uint8)
                
                # Add all instances of this mask type to the collated mask
                for obj_id, seg_data in storage_data['segmentation_data'].items():
                    if mask_type in seg_data and seg_data[mask_type]:
                        polygon = json.loads(seg_data[mask_type])
                        if polygon:
                            binary_mask = polygon2mask(polygon, image_shape)
                            collated_mask[binary_mask > 0] = 1
                
                if np.any(collated_mask):
                    collated_masks[mask_type] = collated_mask
            
            if not collated_masks:
                show_warning("No masks found for segmentation. Skipping mask export.")
                return

            collated_mask_file_path = export_path / f"{self.label_layer_name}_collated_mask.npy"
            np.save(collated_mask_file_path, collated_masks)

    def _get_features_from_layers(self, label_layer_list):
        """Get features from the selected label layers as pd.DataFrame"""
        features = {}
        cur_total_size = 0
        for label_layer_name in label_layer_list:
            if not label_layer_name in self.viewer.layers:
                raise RuntimeError(f"Label layer {label_layer_name} not found in viewer")
            cur_layer = self.viewer.layers[label_layer_name]
            new_total_size = cur_total_size
            for feature in cur_layer.properties.keys():
                if feature not in features:
                    features[feature] = [None for i in range(cur_total_size)]
                features[feature].extend(cur_layer.properties[feature])
                new_total_size = max(new_total_size, len(features[feature]))
            for feature in features.keys():
                if len(features[feature]) < new_total_size:
                    features[feature].extend([None] * (new_total_size - len(features[feature])))
            cur_total_size = new_total_size
        try:
            features_df = pd.DataFrame(features)
        except ValueError as e:
            show_error(f"Error creating DataFrame from features: {e}")
            return None
        return features_df

    def _export_features(self, label_layer, export_path: Path, selected_features, timelapse_export):
        """Export selected features to CSV"""
        # Extract only the selected features
        features_to_export = {}
        if timelapse_export:
            
            if not label_layer.name.startswith("TL_Frame"):
                raise RuntimeError("Internal error: Timelapse checkbox is checked but current layer is not a timelapse frame.")
            
            timelapse_name = get_timelapse_name(label_layer.name)

            if timelapse_name not in self.timelapses:
                raise RuntimeError(f"Timelapse '{timelapse_name}' not found.")
            
            timelapse_layers = self.timelapses[timelapse_name]
            filename = f"{timelapse_name}_features.csv"
            for feature in selected_features:
                features_to_export[feature] = []
            features_to_export['frame_idx'] = []

            for label_layer_name in timelapse_layers:
                if not label_layer_name in self.viewer.layers:
                    raise RuntimeError(f"Label layer {label_layer_name} not found in viewer")
                if not label_layer_name.startswith("TL_Frame"):
                    raise RuntimeError(f"Layer {label_layer_name} is in timelapse but not a timelapse frame")
                frame_layer = self.viewer.layers[label_layer_name]
                frame_idx = int(label_layer_name.split('_')[1][5:])
                feature_sizes = {}
                for feature in selected_features:
                    if feature in frame_layer.properties:
                        features_to_export[feature].extend(frame_layer.properties[feature])
                        feature_sizes[feature] = len(frame_layer.properties[feature])
                    else:
                        feature_sizes[feature] = 0
                uniform_size = max(feature_sizes.values())
                features_to_export['frame_idx'].extend([frame_idx] * uniform_size)
                for feature, cur_size in feature_sizes.items():
                    if cur_size < uniform_size:
                        show_warning(f"Missing data for feature {feature} in layer {label_layer_name}. Extending with None...")
                        features_to_export[feature].extend([None] * (uniform_size - cur_size))
        else:
            filename = f"{label_layer.name}_features.csv"
            for feature in selected_features:
                if feature in label_layer.properties:
                    features_to_export[feature] = label_layer.properties[feature]
                else:
                    show_warning(f"Feature '{feature}' not found in the layer properties.")
        
        # Convert to pandas DataFrame
        if features_to_export:
            df = pd.DataFrame(features_to_export)
            features_file_path = export_path / filename
            df.to_csv(features_file_path, index=False)
        else:
            show_warning("No features selected for export or no features available.")

    def _on_model_selection_changed(self):
        """ Is called when user selects a new model from the dropdown menu. """
        self.model_name = self.model_selection.currentText()

    def _on_choose_model_clicked(self):
        """ Is called whenever browse button is clicked for model selection """
        # called when the user hits the 'browse' button to select a model
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.AnyFile)
        if fd.exec():
            model_path = fd.selectedFiles()[0]
        shutil.copy2(model_path, settings.MODELS_DIR)
        model_name = utils.add_to_dict(model_path)
        self.model_selection.addItem(model_name)

    def _on_window_sizes_changed(self):
        """ Is called whenever user changes the window sizes text box """
        new_window_sizes = self.window_sizes_textbox.text()
        new_window_sizes = new_window_sizes.split(',')
        self.window_sizes = [int(win_size) for win_size in new_window_sizes]

    def _on_downsampling_changed(self):
        """ Is called whenever user changes the downsampling text box """
        new_downsampling = self.downsampling_textbox.text()
        new_downsampling = new_downsampling.split(',')
        self.downsampling = [int(ds) for ds in new_downsampling]

    def _rerun(self):
        """ Is called whenever user changes one of the two parameter sliders """
        # check if OrganoiDL instance exists - create it if not and set there current boxes, scores and ids
        if not self.cur_shapes_layer:
            raise ValueError("No current shapes layer found for rerun.")    
        if 'napari-organoid-analyzer:_rerun' in self.cur_shapes_layer.metadata:
            return
        if self.organoiDL.img_scale[0]==0: self.organoiDL.set_scale(self.cur_shapes_layer.scale)

        # make sure to add info to cur_shapes_layer.metadata to differentiate this action from when user adds/removes boxes
        with utils.set_dict_key( self.cur_shapes_layer.metadata, 'napari-organoid-analyzer:_rerun', True):
            # first update bboxes in organoiDLin case user has added/removed
            if self.apply_to_timelapse_checkbox.isVisible() and self.apply_to_timelapse_checkbox.isChecked():
                if not self.cur_shapes_layer.name.startswith("TL_Frame"):
                    raise RuntimeError("Internal error: Timelapse checkbox is checked but current layer is not a timelapse frame.")
                timelapse_name = get_timelapse_name(self.cur_shapes_layer.name)
                if timelapse_name not in self.timelapses or self.cur_shapes_layer.name not in self.timelapses[timelapse_name]:
                    raise RuntimeError(f"Internal error: unknown timelapse or frame name {timelapse_name}")
                old_shape_layer_name = self.cur_shapes_layer.name
                if not old_shape_layer_name in self.label2im:
                    raise RuntimeError(f"Internal error: no image layer found for {old_shape_layer_name}")
                image_layer_name = self.label2im[old_shape_layer_name]
                if not image_layer_name in self.viewer.layers:
                    raise RuntimeError(f"Internal error: image layer {image_layer_name} not found in viewer")
                
                image_shape = self.viewer.layers[image_layer_name].data.shape[1:3]
                for frame_name in self.timelapses[timelapse_name]:
                    self.organoiDL.update_bboxes_scores(frame_name,
                                                self.viewer.layers[frame_name].data, 
                                                self.viewer.layers[frame_name].properties,
                                                image_shape
                                            )
                    self._update_detections(frame_name)
                self.cur_shapes_layer = self.viewer.layers[old_shape_layer_name]
                self._update_num_organoids(len(self.cur_shapes_layer.data))
                self._update_cur_shapes_layer_label_and_checkbox()
            else:
                if not self.cur_shapes_layer.name in self.label2im:
                    raise RuntimeError(f"Internal error: no image layer found for {self.cur_shapes_layer.name}")
                image_layer_name = self.label2im[self.cur_shapes_layer.name]
                if not image_layer_name in self.viewer.layers:
                    raise RuntimeError(f"Internal error: image layer {image_layer_name} not found in viewer")
                image_shape = self.viewer.layers[image_layer_name].data.shape[:2]
                self.organoiDL.update_bboxes_scores(self.cur_shapes_layer.name,
                                                self.cur_shapes_layer.data, 
                                                self.cur_shapes_layer.properties,
                                                image_shape
                                            )
                self._update_detections(self.cur_shapes_layer.name)

    def _on_diameter_slider_changed(self):
        """ Is called whenever user changes the Minimum Diameter slider """
        # get current value
        if self.diameter_textbox_changed: return
        self.min_diameter = self.min_diameter_slider.value()
        if self.cur_shapes_layer is not None:
            self.stored_diameters[self.cur_shapes_layer.name] = self.min_diameter
        self.diameter_slider_changed = True
        if int(self.min_diameter_textbox.text())!= self.min_diameter:
            self.min_diameter_textbox.setText(str(self.min_diameter))
        self.diameter_slider_changed = False
        # check if no labels loaded yet
        if len(self.shape_layer_names)==0: return
        self._rerun() 
    
    def _on_diameter_textbox_changed(self):
        """ Is called whenever user changes the minimum diameter from the textbox """
        # check if no labels loaded yet
        if self.diameter_slider_changed: return
        self.min_diameter = int(self.min_diameter_textbox.text())
        self.diameter_textbox_changed = True
        if self.cur_shapes_layer is not None:
            self.stored_diameters[self.cur_shapes_layer.name] = self.min_diameter
        if self.min_diameter_slider.value() != self.min_diameter:
            self.min_diameter_slider.setValue(self.min_diameter)
        self.diameter_textbox_changed = False
        if len(self.shape_layer_names)==0: return
        self._rerun()

    def _on_confidence_slider_changed(self):
        """ Is called whenever user changes the confidence slider """
        if self.confidence_textbox_changed: return
        self.confidence = self.confidence_slider.value()/100
        self.confidence_slider_changed = True
        if self.cur_shapes_layer is not None:
            self.stored_confidences[self.cur_shapes_layer.name] = self.confidence
        if float(self.confidence_textbox.text()) != self.confidence:
            self.confidence_textbox.setText(str(self.confidence))
        self.confidence_slider_changed = False
        # check if no labels loaded yet
        if len(self.shape_layer_names)==0: return
        self._rerun()

    def _on_confidence_textbox_changed(self):
        """ Is called whenever user changes the confidence value from the textbox """
        if self.confidence_slider_changed: return
        self.confidence = float(self.confidence_textbox.text())
        slider_conf_value = int(self.confidence*100)
        self.confidence_textbox_changed = True
        if self.cur_shapes_layer is not None:
            self.stored_confidences[self.cur_shapes_layer.name] = self.confidence
        if self.confidence_slider.value() != slider_conf_value:
            self.confidence_slider.setValue(slider_conf_value)
        self.confidence_textbox_changed = False
        if len(self.shape_layer_names)==0: return
        self._rerun()

    def _on_image_selection_changed(self):
        """ Is called whenever a new image has been selected from the drop down box """
        self.image_layer_name = self.image_layer_selection.currentText()
    
    def _on_reset_click(self):
        """ Is called whenever Reset Configs button is clicked """
        # reset params
        self.min_diameter = 30
        self.confidence = 0.8
        vis_confidence = int(self.confidence*100)
        self.min_diameter_slider.setValue(self.min_diameter)
        self.confidence_slider.setValue(vis_confidence)
        if self.image_layer_name:
            # reset to original image
            self.viewer.layers[self.image_layer_name].data = self.original_images[self.image_layer_name]
            self.viewer.layers[self.image_layer_name].contrast_limits = self.original_contrast[self.image_layer_name]

    def _on_screenshot_click(self):
        """ Is called whenever Take Screenshot button is clicked """
        screenshot=self.viewer.screenshot()
        if not self.image_layer_name: potential_name = datetime.now().strftime("%d%m%Y%H%M%S")+'screenshot.png'
        else: potential_name = self.image_layer_name+datetime.now().strftime("%d%m%Y%H%M%S")+'_screenshot.png'
        fd = QFileDialog()
        name,_ = fd.getSaveFileName(self, 'Save File', potential_name, 'Image files (*.png);;(*.tiff)') #, 'CSV Files (*.csv)')
        if name: imsave(name, screenshot)

    def _on_custom_labels_click(self):
        """
        Called when user clicks on button to add custom organoid annotation to image
        """
        
        if not self.guided_mode:
            if not self.image_layer_name: 
                show_error('Cannot assign custom label to image. Please load an image first!')
                return

            img_data = self.viewer.layers[self.image_layer_name].data
            loaded_cached_data = self.compute_and_check_image_hash(img_data, self.image_layer_name)
            if loaded_cached_data:
                return
            
            if self.organoiDL.img_scale[0] == 0:
                self.organoiDL.set_scale(self.viewer.layers[self.image_layer_name].scale[:2])

            new_layer_name = f'{self.image_layer_name}-Labels-Custom-{datetime.strftime(datetime.now(), "%H_%M_%S")}'
            img_data = self.viewer.layers[self.image_layer_name].data

            if self.image_layer_name in self.timelapse_image_layers:
                # Add custom labels for timelapse
                timelapse_name = f'{self.image_layer_name}-Labels-Custom-'

                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Create Custom Labels for Timelapse")
                msg_box.setText("Do you want to create custom labels for all frames in the timelapse?")
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                reply = msg_box.exec()
                if reply == QMessageBox.Yes:
                    # Create a labels layer for each frame
                    for i in range(img_data.shape[0]):
                        frame_layer_name = f'TL_Frame{i}_{timelapse_name}'
                        if not frame_layer_name in self.timelapses[timelapse_name]:
                            self.organoiDL.update_bboxes_scores(
                                frame_layer_name,
                                np.array([]),
                                {'bbox_id': [], 'score': []},
                                img_data.shape[1:3]
                            )
                            self._update_detections(frame_layer_name, confidence=0.8, min_diameter=30, image_layer_name=self.image_layer_name)
                    curr_step = list(self.viewer.dims.current_step)
                    curr_step[0] = img_data.shape[0] - 1
                    self.viewer.dims.current_step = tuple(curr_step)
                else:
                    # Create a labels layer for the current frame only
                    if not hasattr(self.viewer.dims, "current_step") or len(self.viewer.dims.current_step) == 0:
                        show_error("Internal error: Unable to determine current frame index.")
                        return
                    cur_frame = self.viewer.dims.current_step[0]
                    if cur_frame >= img_data.shape[0]:
                        show_error(f"Current frame index {cur_frame} exceeds the number of frames in the image.")
                        return
                    frame_layer_name = f'TL_Frame{cur_frame}_{timelapse_name}'
                    if frame_layer_name in self.timelapses[timelapse_name]:
                        show_warning(f"Layer '{frame_layer_name}' already exists.")
                        return
                    self.organoiDL.update_bboxes_scores(
                        frame_layer_name,
                        np.array([]),
                        {'bbox_id': [], 'score': []},
                        img_data.shape[1:3]
                    )
                    self._update_detections(frame_layer_name, confidence=0.8, min_diameter=30, image_layer_name=self.image_layer_name)
            else:
                self.organoiDL.update_bboxes_scores(
                    new_layer_name,
                    np.array([]),
                    {'bbox_id': [], 'score': []},
                    img_data.shape[:2]
                )
                self._update_detections(new_layer_name, confidence=0.8, min_diameter=30, image_layer_name=self.image_layer_name)

        else:
            new_layer_name = f'Guidance-{datetime.strftime(datetime.now(), "%H_%M_%S")}'
            self.guidance_layers.add(new_layer_name)
            properties = {}
            text_params = {}
            edge_color = settings.COLOR_CLASS_1
            self.viewer.add_shapes( 
                    name=new_layer_name,
                    scale=self.viewer.layers[self.image_layer_name].scale[:2],
                    face_color='transparent',  
                    properties = properties,
                    text = text_params,
                    edge_color=edge_color,
                    shape_type='rectangle',
                    edge_width=12
            )

    def _update_added_image(self, added_items):
        """
        Update the selection box with new images if images have been added and update the self.original_images and self.original_contrast dicts.
        Set the latest added image to the current working image (self.image_layer_name)
        """
        for layer_name in added_items:
            self.image_layer_names.append(layer_name)
            if not layer_name.startswith('Segmentation-') and not layer_name.startswith('TL_'):
                #try:
                self._preprocess(layer_name, self.viewer.layers[layer_name].data)
                image_data = self.viewer.layers[layer_name].data
                if image_data.ndim == 4:
                    self.timelapse_image_layers.add(layer_name)
                    timelapse_name = f'{layer_name}-Labels-Cache-{datetime.strftime(datetime.now(), "%H_%M_%S")}'
                    for i in range(image_data.shape[0]):
                        shapes_name = f'TL_Frame{i}_{timelapse_name}'
                        self.compute_and_check_image_hash(image_data[i], layer_name, shapes_name)
                    self._on_frame_change()
                elif image_data.ndim == 3 or image_data.ndim == 2:
                    self.compute_and_check_image_hash(image_data, layer_name)
                else:
                    show_error(f"Unsupported image format for layer {layer_name}: shape {image_data.shape}")
                    continue
                self.remember_choice_for_image_import = None
                self.image_layer_selection.addItem(layer_name)
                self.image_layer_name = layer_name
                self.image_layer_selection.setCurrentText(self.image_layer_name)

            self.original_images[layer_name] = self.viewer.layers[layer_name].data
            self.original_contrast[layer_name] = self.viewer.layers[self.image_layer_name].contrast_limits

    def _update_removed_image(self, removed_layers):
        """
        Update the selection box by removing image names if image has been deleted and remove items from self.original_images and self.original_contrast dicts.
        """
        # update drop-down selection box and remove image from dict
        for removed_layer in removed_layers:
            item_id = self.image_layer_selection.findText(removed_layer)
            if removed_layer in self.timelapse_image_layers:
                self.timelapse_image_layers.remove(removed_layer)
            if item_id >= 0:
                self.image_layer_selection.removeItem(item_id)
            self.original_images.pop(removed_layer, None)
            self.original_contrast.pop(removed_layer, None)
            self.im2signal.pop(removed_layer, None)
        signal_dict = {key: val for key, val in self.im2signal.items() if not val in removed_layers}
        self.im2signal = signal_dict
        self._on_labels_layer_change()

    def _update_added_shapes(self, added_items):
        """
        Update the selection box by shape layer names if it they have been added, update current working shape layer and instantiate OrganoiDL if not already there
        """
        # update the drop down box displaying shape layer names for saving

        for layer_name in added_items:
            self.shape_layer_names.append(layer_name)
            if layer_name in self.guidance_layers:
                self.guidance_selection.addItem(layer_name)
                self.guidance_layer_name = layer_name
                self.guidance_selection.setCurrentText(self.guidance_layer_name)
            else:
                self.segmentation_image_layer_selection.addItem(layer_name)
                self.annotation_image_layer_selection.addItem(layer_name)
                self.cur_shapes_layer = self.viewer.layers[layer_name]
                self._update_num_organoids(len(self.cur_shapes_layer.data))
                self.cur_shapes_layer.events.data.connect(self.shapes_event_handler)
                self.cur_shapes_layer.events.highlight.connect(self._on_shape_selected)
                self.cur_shapes_layer.events.name.connect(self._on_layer_name_change)
                # update label and checkbox for current shapes layer
                self._update_cur_shapes_layer_label_and_checkbox()
        
    def _update_remove_shapes(self, removed_layers):
        """
        Update the selection box by removing shape layer names if it they been deleted and set 
        """
        # update selection box by removing image names if image has been deleted       
        for removed_name in removed_layers:
            if removed_name in self.guidance_layers:
                self.guidance_layers.remove(removed_name)
                item_id = self.guidance_selection.findText(removed_name)
                if item_id >= 0:
                    self.guidance_selection.removeItem(item_id)
                if removed_name == self.guidance_layer_name:
                    self.guidance_layer_name = None
                    self.guidance_selection.setCurrentText("None")
            else:
                item_id = self.segmentation_image_layer_selection.findText(removed_name)
                self.segmentation_image_layer_selection.removeItem(item_id)
                item_id = self.annotation_image_layer_selection.findText(removed_name)
                self.annotation_image_layer_selection.removeItem(item_id)
                self.label2im.pop(removed_name, None)
                self.stored_confidences.pop(removed_name, None)
                self.stored_diameters.pop(removed_name, None)

                if removed_name.startswith('TL_Frame'):
                    timelapse_name = get_timelapse_name(removed_name)
                    if timelapse_name in self.timelapses:
                        self.timelapses[timelapse_name].remove(removed_name)
                        if len(self.timelapses[timelapse_name]) == 0:
                            if self.cur_timelapse_name == timelapse_name:
                                self.cur_timelapse_name = None
                            del self.timelapses[timelapse_name]
                            self.timelapse_segmentations.pop(timelapse_name, None)
                            item_id = self.timelapse_selection.findText(timelapse_name)
                            if item_id >= 0:
                                self.timelapse_selection.removeItem(item_id)
                    else:
                        show_error(f"Corresponding timelapse '{timelapse_name}' not found.")

                if self.cur_shapes_layer and removed_name==self.cur_shapes_layer.name: 
                    self._update_num_organoids(0)
                    self.cur_shapes_layer = None
                    self._update_cur_shapes_layer_label_and_checkbox()
                self.organoiDL.remove_shape_from_dict(removed_name)

    def shapes_event_handler(self, event):
        """
        This function will be called every time the current shapes layer data changes
        """
        # Do not perform update if changes in the layer are due to slider changes or tracking
        if (
            'napari-organoid-analyzer:_rerun' in self.cur_shapes_layer.metadata or 
            'napari-organoid-analyzer:_tracking' in self.cur_shapes_layer.metadata or 
            'napari-organoid-analyzer:_shape_handler' in self.cur_shapes_layer.metadata
        ):
            return
        
        
        with utils.set_dict_key(self.cur_shapes_layer.metadata, 'napari-organoid-analyzer:_shape_handler', True):
        # get new ids, new boxes and update the number of organoids
            if not self.cur_shapes_layer.name in self.label2im:
                raise RuntimeError(f"Internal error: no image layer found for {self.cur_shapes_layer.name}")
            image_layer_name = self.label2im[self.cur_shapes_layer.name]
            if not image_layer_name in self.viewer.layers:
                raise RuntimeError(f"Internal error: image layer {image_layer_name} not found in viewer")
            image_data = self.viewer.layers[image_layer_name].data
            if image_data.ndim == 4:
                image_shape = self.viewer.layers[image_layer_name].data.shape[1:3]
            else:
                image_shape = self.viewer.layers[image_layer_name].data.shape[:2]
            new_bboxes = self.cur_shapes_layer.data
            properties = self.cur_shapes_layer.properties.copy()
            new_ids = properties.get('bbox_id', [])
            scores = properties.get('score', [1.0]*len(new_ids))
            self._update_num_organoids(len(new_ids))
            curr_next_id = self.organoiDL.storage[self.cur_shapes_layer.name]['next_id']
        
            # check if duplicate ids
            contains_nan = False
            try:
                new_ids = [int(id_val) for id_val in new_ids]
            except ValueError:
                contains_nan = True

            if len(new_ids) > len(set(new_ids)) or contains_nan:
                # Duplicate or missing IDs for some detections.
                used_id = set()
                for idx, id_val in enumerate(new_ids):
                    curr_nan = False
                    try:
                        id_val = int(id_val)
                    except ValueError:
                        curr_nan = True
                    if id_val in used_id or curr_nan:
                        new_ids[idx] = int(curr_next_id)
                        used_id.add(curr_next_id)
                        curr_next_id += 1
                        scores[idx] = 1.0
                    else:
                        used_id.add(id_val)


            new_ids = list(map(int, new_ids))
            properties['bbox_id'] = new_ids
            properties['score'] = scores
            self.cur_shapes_layer.properties = properties
            self.organoiDL.update_bboxes_scores(self.cur_shapes_layer.name, new_bboxes, properties, image_shape)
            self._save_cache_results(self.cur_shapes_layer.name)
            #self._update_detections(self.cur_shapes_layer.name, image_layer_name)
            bboxes, properties = self.organoiDL.apply_params(
                self.cur_shapes_layer.name,
                self.stored_confidences.get(self.cur_shapes_layer.name, self.confidence),
                self.stored_diameters.get(self.cur_shapes_layer.name, self.min_diameter)
            )
            self.cur_shapes_layer.properties = properties
            self._update_detection_data_tab()
            self.cur_shapes_layer.refresh()
            self.cur_shapes_layer.refresh_text()

    def _setup_input_widget(self):
        """
        Sets up the GUI part which corresponds to the input configurations
        """
        # setup all the individual boxes
        input_box = self._setup_input_box()
        guidance_box = self._setup_guidance_box()  # Add guidance selector
        model_box = self._setup_model_box()
        window_sizes_box = self._setup_window_sizes_box()
        downsampling_box = self._setup_downsampling_box()
        run_box = self._setup_run_box()
        self._setup_progress_box()

        # and add all these to the layout
        input_widget = QGroupBox('Input configurations')
        vbox = QVBoxLayout()
        vbox.addLayout(input_box)
        vbox.addLayout(guidance_box)  # Add guidance selector to layout
        vbox.addLayout(model_box)
        vbox.addLayout(window_sizes_box)
        vbox.addLayout(downsampling_box)
        vbox.addLayout(run_box)
        vbox.addWidget(self.progress_box)
        input_widget.setLayout(vbox)
        return input_widget

    def _setup_guidance_box(self):
        """
        Sets up the GUI part where the guidance type is selected.
        """
        hbox = QHBoxLayout()
        # setup label
        guidance_label = QLabel('Guidance layer: ', self)
        guidance_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # setup drop down option for selecting guidance type
        self.guidance_selection = QComboBox()
        self.guidance_selection.addItem('None')
        
        self.guidance_selection.currentIndexChanged.connect(self._on_guidance_selection_changed)
        # and add all these to the layout
        hbox.addWidget(guidance_label, 2)
        hbox.addWidget(self.guidance_selection, 4)
        return hbox

    def _on_guidance_selection_changed(self, index):
        """
        Callback for when the guidance selection changes.
        """
        if self.guidance_selection.currentText() == 'None':
            self.guidance_layer_name = None
        else:
            self.guidance_layer_name = self.guidance_selection.currentText()

    def _setup_output_widget(self):
        """
        Sets up the GUI part which corresposnds to the parameters and outputs
        """
        # setup all the individual boxes
        self.organoid_number_label = QLabel('Number of organoids: '+str(self.num_organoids), self)
        self.organoid_number_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.cur_shapes_layer_label = QLabel('Modified labels layer: None', self)
        self.cur_shapes_layer_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.apply_to_timelapse_checkbox = QCheckBox("Apply to entire timelapse")
        self.apply_to_timelapse_checkbox.setChecked(False)
        self.apply_to_timelapse_checkbox.setVisible(False)
        # and add all these to the layout
        output_widget = QGroupBox('Parameters and outputs')
        vbox = QVBoxLayout()
        # Add current shapes layer label and timelapse checkbox above sliders
        vbox.addWidget(self.cur_shapes_layer_label)
        vbox.addLayout(self._setup_min_diameter_box())
        vbox.addLayout(self._setup_confidence_box() )
        vbox.addWidget(self.apply_to_timelapse_checkbox)
        vbox.addWidget(self.organoid_number_label)
        vbox.addLayout(self._setup_reset_box())
        
        output_widget.setLayout(vbox)
        return output_widget

    def _setup_input_box(self):
        """
        Sets up the GUI part where the input image is defined
        """
        #self.input_box = QGroupBox()
        hbox = QHBoxLayout()
        # setup label
        image_label = QLabel('Image: ', self)
        image_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # setup drop down option for selecting which image to process
        self.image_layer_selection = QComboBox()
        if self.image_layer_names is not None:
            for name in self.image_layer_names: 
                if not name.startswith('Segmentation-') and not name.startswith('TL_'):
                    self.image_layer_selection.addItem(name)
        #self.image_layer_selection.setItemText(self.image_layer_name)
        self.image_layer_selection.currentIndexChanged.connect(self._on_image_selection_changed)
        # and add all these to the layout
        hbox.addWidget(image_label, 2)
        hbox.addWidget(self.image_layer_selection, 4)
        return hbox

    def _setup_model_box(self):
        """
        Sets up the GUI part where the model is selected from a drop down menu.
        """
        hbox = QHBoxLayout()
        # setup the label
        model_label = QLabel('Model: ', self)
        model_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        # setup the browse files button
        fileOpenButton = QPushButton('Add custom model', self)
        fileOpenButton.show()
        fileOpenButton.clicked.connect(self._on_choose_model_clicked)
        
        # setup drop down option for selecting which image to process
        self.model_selection = QComboBox()
        for name in settings.MODELS.keys(): self.model_selection.addItem(name)
        self.model_selection.setCurrentIndex(self.model_id)
        self.model_selection.currentIndexChanged.connect(self._on_model_selection_changed)
        
        # and add all these to the layout
        hbox.addWidget(model_label, 2)
        hbox.addWidget(self.model_selection, 4)
        hbox.addWidget(fileOpenButton, 4)
        return hbox

    def _setup_window_sizes_box(self):
        """
        Sets up the GUI part where the window sizes parameters are set
        """
        #self.window_sizes_box = QGroupBox()
        hbox = QHBoxLayout()
        info_text = ("Typically a ratio of 512 to 1 between window size and downsampling rate will give good results, (larger window \n"
                    "sizes can lead to a drop in performance). Note that small window sizes will signicantly impact the runtime of the \n"
                    "algorithm. For organoids of different sizes consider setting multiple windows sizes. Hit Enter for the change to \n"
                    "take effect.")
        # setup label
        window_sizes_label = QLabel('Window sizes: [size1, size2, ...]', self)
        window_sizes_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        window_sizes_label.setToolTip(info_text)
        # setup textbox
        self.window_sizes_textbox = QLineEdit(self)
        text = [str(window_size) for window_size in self.window_sizes]
        text = ','.join(text)
        self.window_sizes_textbox.setText(text)
        self.window_sizes_textbox.returnPressed.connect(self._on_window_sizes_changed)
        self.window_sizes_textbox.setToolTip(info_text)
        # and add all these to the layout
        hbox.addWidget(window_sizes_label)
        hbox.addWidget(self.window_sizes_textbox)   
        #self.window_sizes_box.setLayout(hbox)   
        #self.window_sizes_box.setStyleSheet("border: 0px")  
        return hbox
    
    def _setup_downsampling_box(self):
        """
        Sets up the GUI part where the downsampling parameters are set
        """
        #self.downsampling_box = QGroupBox()
        hbox = QHBoxLayout()
        info_text = ("To detect large organoids (and ignore smaller structures) you can increase the downsampling rate. \n"
                    "If your organoids are small and are being missed by the algorithm, consider reducing the downsampling\n"
                    "rate. The number of downsampling inputs should match the number of windows sizes. Hit Enter for the \n"
                    "change to take effect. See window sizes for more info.")

        # setup label
        downsampling_label = QLabel('Downsampling: [ds1, ds2, ...]', self)
        downsampling_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        downsampling_label.setToolTip(info_text)
        # setup textbox
        self.downsampling_textbox = QLineEdit(self)
        text = [str(ds) for ds in self.downsampling]
        text = ','.join(text)
        self.downsampling_textbox.setText(text)
        self.downsampling_textbox.returnPressed.connect(self._on_downsampling_changed)
        self.downsampling_textbox.setToolTip(info_text)
        # and add all these to the layout
        hbox.addWidget(downsampling_label)
        hbox.addWidget(self.downsampling_textbox) 
        #self.downsampling_box.setLayout(hbox)
        #self.downsampling_box.setStyleSheet("border: 0px") 
        return hbox

    def _setup_run_box(self):
        """
        Sets up the GUI part where the user hits the run button
        """
        vbox = QVBoxLayout()
        
        # Button layout
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        run_btn = QPushButton("Run Organoid Counter")
        run_btn.clicked.connect(self._on_run_click)
        run_btn.setStyleSheet("border: 0px")
        import_btn = QPushButton("Import detections")
        import_btn.setStyleSheet("border: 0px")
        import_btn.clicked.connect(self._on_import_detections_click)
        hbox.addWidget(run_btn)
        hbox.addWidget(import_btn)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        # Custom labels and detection guidance
        hbox_custom = QHBoxLayout()
        custom_btn = QPushButton("Add custom labels")
        custom_btn.clicked.connect(self._on_custom_labels_click)
        custom_btn.setStyleSheet("border: 0px")
        self.detection_guidance_checkbox = QCheckBox("Detection guidance")
        self.detection_guidance_checkbox.setChecked(False)
        self.detection_guidance_checkbox.stateChanged.connect(self._on_detection_guidance_checkbox_changed)
        hbox_custom.addStretch(1)
        hbox_custom.addWidget(custom_btn)
        hbox_custom.addSpacing(15)
        hbox_custom.addWidget(self.detection_guidance_checkbox)
        hbox_custom.addStretch(1)
        vbox.addLayout(hbox_custom)
        
        # Cache checkbox
        cache_hbox = QHBoxLayout()
        cache_hbox.addStretch(1)
        self.cache_checkbox = QCheckBox("Cache results")
        self.cache_checkbox.setChecked(self.cache_enabled)
        self.cache_checkbox.stateChanged.connect(self._on_cache_checkbox_changed)
        cache_hbox.addWidget(self.cache_checkbox)
        cache_hbox.addStretch(1)
        vbox.addLayout(cache_hbox)
        
        return vbox

    def _on_import_detections_click(self):
        """
        Called when the Import Detections button is pressed.
        """
        if not self.image_layer_selection.currentText():
            show_warning("No corresponding image layer")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Detections", "", "JSON files (*.json)")
        if not file_path:
            return
        try:
            with open(file_path, "r") as f:
                detection_data = json.load(f)
                if not "type" in detection_data:
                    raise ValueError("Invalid detection data format. Expected 'type' key with value 'layer' or 'timelapse'.")
                if detection_data["type"] == "timelapse":
                    for frame_name, frame_data in detection_data.items():
                        if "detection_data" in frame_data:
                            frame_data['detection_data'] = {
                                int(k): v for k, v in frame_data['detection_data'].items()
                            }
                        if "segmentation_data" in frame_data:
                            frame_data['segmentation_data'] = {
                                int(k): v for k, v in frame_data.get('segmentation_data', {}).items()
                            }
                elif detection_data["type"] == "layer":
                    if "detection_data" in detection_data:
                        detection_data['detection_data'] = {
                            int(k): v for k, v in detection_data['detection_data'].items()
                        }
                    if "segmentation_data" in detection_data:
                        detection_data['segmentation_data'] = {
                            int(k): v for k, v in detection_data.get('segmentation_data', {}).items()
                        }
                else:
                    raise ValueError("Invalid detection data type. Expected 'layer' or 'timelapse'.")
        except Exception as e:
            show_error(f"Failed to load detections: {e}")
            return

        data_type = detection_data.pop("type", None)
        image_layer_name = self.image_layer_selection.currentText()
        if not image_layer_name in self.viewer.layers:
            show_error("No corresponding image layer found in the viewer")
            return
        image_data = self.viewer.layers[image_layer_name].data
        image_layer_shape = self.viewer.layers[image_layer_name].data.shape

        if self.organoiDL.img_scale[0] == 0:
            self.organoiDL.set_scale(self.viewer.layers[image_layer_name].scale[:2])

        if data_type == "layer":
            # If the imported data is for a single layer
            if list(image_layer_shape[:2]) != detection_data["image_size"]:
                show_error(f"Import failed! Image size mismatch. current: {image_layer_shape[:2]}, imported: {detection_data['image_size']}")
                return
            labels_layer_name = f"{image_layer_name}-Labels-Imported-{datetime.strftime(datetime.now(), '%H_%M_%S')}"
            confidence = detection_data.pop('confidence', self.confidence)
            min_diameter = detection_data.pop('min_diameter', self.min_diameter)
            self.organoiDL.storage[labels_layer_name] = detection_data
            self._update_detections(labels_layer_name, confidence=confidence, min_diameter=min_diameter, image_layer_name=image_layer_name)
        elif data_type == "timelapse":
            # If the imported data is for a timelapse
            if len(image_layer_shape) < 3:
                show_error("Import failed! Selected image layer is not a timelapse.")
                return
            cur_total_frames = image_layer_shape[0]
            cur_image_shape = list(image_layer_shape[1:3])
            timelapse_name = f"{image_layer_name}-Labels-Imported-{datetime.strftime(datetime.now(), '%H_%M_%S')}"
            for frame_idx, frame_data in detection_data.items():
                if int(frame_idx) >= cur_total_frames:
                    show_warning(f"Skipping frame {frame_idx} as it exceeds the current timelapse length ({cur_total_frames} frames).")
                    continue
                if frame_data["image_size"] != cur_image_shape:
                    show_warning(f"Skipping frame {frame_idx} due to image size mismatch. Current: {cur_image_shape}, Imported: {frame_data['image_size']}")
                    continue
                frame_name = f"TL_Frame{frame_idx}_{timelapse_name}"
                confidence = frame_data.pop('confidence', self.confidence)
                min_diameter = frame_data.pop('min_diameter', self.min_diameter)
                self.organoiDL.storage[frame_name] = frame_data
                self._update_detections(frame_name, confidence=confidence, min_diameter=min_diameter, image_layer_name=image_layer_name)


    def _on_cache_checkbox_changed(self, state):
        """Called when cache checkbox is toggled"""
        self.cache_enabled = (state == Qt.Checked)

    def _on_detection_guidance_checkbox_changed(self, state):
        """
        Called when the detection guidance checkbox is toggled.
        """
        self.guided_mode = (state == Qt.Checked)

    def _setup_progress_box(self):
        """
        Sets up the GUI part which appears when the model is being downloaded.
        This should only happen once for each model whihc is then stored in cache. 
        """
        self.progress_box = QGroupBox()
        hbox = QHBoxLayout()
        download_label = QLabel('Downloading model progress: ', self)
        download_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.progress_bar = QProgressBar(self) # creating progress bar
        hbox.addWidget(download_label)
        hbox.addWidget(self.progress_bar)
        self.progress_box.setLayout(hbox)
        self.progress_box.hide()

    def _setup_min_diameter_box(self):
        """
        Sets up the GUI part where the minimum diameter parameter is displayed
        """
        hbox = QHBoxLayout()
        # setup the min diameter slider
        self.min_diameter_slider = QSlider(Qt.Horizontal)
        self.min_diameter_slider.setMinimum(10)
        self.min_diameter_slider.setMaximum(100)
        self.min_diameter_slider.setSingleStep(10)
        self.min_diameter_slider.setValue(self.min_diameter)
        self.min_diameter_slider.valueChanged.connect(self._on_diameter_slider_changed)
        # setup the label
        min_diameter_label = QLabel('Minimum Diameter [um]: ', self)
        min_diameter_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # setup text box
        self.min_diameter_textbox = QLineEdit(self)
        self.min_diameter_textbox.setText(str(self.min_diameter))
        self.min_diameter_textbox.returnPressed.connect(self._on_diameter_textbox_changed)  
        # and add all these to the layout
        hbox.addWidget(min_diameter_label, 4)
        hbox.addWidget(self.min_diameter_textbox, 1)
        hbox.addWidget(self.min_diameter_slider, 5)
        return hbox

    def _setup_confidence_box(self):
        """
        Sets up the GUI part where the confidence parameter is displayed
        """
        hbox = QHBoxLayout()
        # setup confidence slider
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(5)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setSingleStep(5)
        vis_confidence = int(self.confidence*100)
        self.confidence_slider.setValue(vis_confidence)
        self.confidence_slider.valueChanged.connect(self._on_confidence_slider_changed)
        # setup label
        confidence_label = QLabel('confidence: ', self)
        confidence_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # setup text box
        self.confidence_textbox = QLineEdit(self)
        self.confidence_textbox.setText(str(self.confidence))
        self.confidence_textbox.returnPressed.connect(self._on_confidence_textbox_changed)  
        # and add all these to the layout
        hbox.addWidget(confidence_label, 3)
        hbox.addWidget(self.confidence_textbox, 1)
        hbox.addWidget(self.confidence_slider, 6)
        return hbox

    def _setup_reset_box(self):
        """
        Sets up the GUI part where screenshot and reset are available to the user
        """
        #self.reset_box = QGroupBox()
        hbox = QHBoxLayout()
        # setup button for resetting parameters
        self.reset_btn = QPushButton("Reset Configs")
        self.reset_btn.clicked.connect(self._on_reset_click)
        # setup button for taking screenshot of current viewer
        self.screenshot_btn = QPushButton("Take screenshot")
        self.screenshot_btn.clicked.connect(self._on_screenshot_click)
        # and add all these to the layout
        hbox.addStretch(1)
        hbox.addWidget(self.screenshot_btn)
        hbox.addSpacing(15)
        hbox.addWidget(self.reset_btn)
        hbox.addStretch(1)
        #self.reset_box.setLayout(hbox)
        #self.reset_box.setStyleSheet("border: 0px")
        return hbox

    def _setup_segmentation_widget(self):
        """
        Sets up the GUI part for segmentation configuration.
        """
        segmentation_widget = QGroupBox('Segmentation configuration')
        vbox = QVBoxLayout()
        
        # Image layer selection
        hbox_img = QHBoxLayout()
        image_label = QLabel('Labels layer: ', self)
        image_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.segmentation_image_layer_selection = QComboBox()
        if self.image_layer_names is not None:
            for name in self.image_layer_names:
                if not name.startswith('Segmentation-') and not name.startswith('TL_'):
                    self.segmentation_image_layer_selection.addItem(name)
        self.segmentation_image_layer_selection.currentIndexChanged.connect(self._on_labels_layer_change)
        hbox_img.addWidget(image_label, 2)
        hbox_img.addWidget(self.segmentation_image_layer_selection, 4)
        vbox.addLayout(hbox_img)
        
        # Run for entire timelapse checkbox
        self.run_for_timelapse_checkbox = QCheckBox("Apply to entire timelapse")
        self.run_for_timelapse_checkbox.setVisible(False)
        vbox.addWidget(self.run_for_timelapse_checkbox)
        
        # Run segmentation button
        hbox_run = QHBoxLayout()
        hbox_run.addStretch(1)
        run_segmentation_btn = QPushButton("Run Segmentation")
        run_segmentation_btn.clicked.connect(self._on_run_segmentation)
        run_segmentation_btn.setStyleSheet("border: 0px")
        export_btn = QPushButton("Export data")
        export_btn.clicked.connect(self._on_export_layer_click)
        export_btn.setStyleSheet("border: 0px")
        hbox_run.addWidget(run_segmentation_btn)
        hbox_run.addSpacing(15)
        hbox_run.addWidget(export_btn)
        hbox_run.addStretch(1)   
        vbox.addLayout(hbox_run)

        signal_hbox = QHBoxLayout()
        signal_hbox.addStretch(1)
        signal_btn = QPushButton("+ Add signal")
        signal_btn.clicked.connect(self._on_add_signal)
        signal_hbox.addWidget(signal_btn)
        signal_hbox.addStretch(1)
        vbox.addLayout(signal_hbox)

        self.signal_layer_label = QLabel("Associated signal layers:")
        self.signal_layer_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.signal_layer_label.setWordWrap(True)
        vbox.addWidget(self.signal_layer_label)

        self.signals_list = QTableWidget()
        self.signals_list.setColumnCount(3)
        self.signals_list.setHorizontalHeaderLabels(["Signal", "Layer", "Delete"])
        self.signals_list.verticalHeader().setVisible(False)
        self.signals_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.signals_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.signals_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.signals_list.verticalHeader().setDefaultSectionSize(50)
        self.signals_list.verticalHeader().setMinimumSectionSize(50)
        self.signals_list.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.signals_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        vbox.addWidget(self.signals_list)

        segmentation_widget.setLayout(vbox)
        return segmentation_widget

    def _setup_search_tool_widget(self):
        """
        Sets up the GUI part for the search tool.
        """
        search_widget = QGroupBox('Search tool')
        vbox = QVBoxLayout()
        # Add current shapes layer label at the top
        self.search_cur_shapes_layer_label = QLabel('Current shapes layer: None', self)
        self.search_cur_shapes_layer_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        vbox.addWidget(self.search_cur_shapes_layer_label)
        hbox = QHBoxLayout()
        label = QLabel("Detection ID:")
        self.search_id_textbox = QLineEdit()
        self.search_id_textbox.setValidator(QIntValidator(1, 2**31-1))  # Only positive integers
        self.search_id_textbox.returnPressed.connect(self._on_find_detection_click)
        find_btn = QPushButton("Find")
        find_btn.clicked.connect(self._on_find_detection_click)
        hbox.addWidget(label)
        hbox.addWidget(self.search_id_textbox)
        hbox.addWidget(find_btn)
        vbox.addLayout(hbox)
        search_widget.setLayout(vbox)
        return search_widget

    def _on_find_detection_click(self):
        """
        Called when the user clicks the 'Find' button in the search tool.
        """
        if not self.cur_shapes_layer:
            show_warning("Please select a shapes layer first.")
            return
        search_id_text = self.search_id_textbox.text()
        if not search_id_text:
            show_warning("Please enter a Detection ID.")
            return
        search_id = int(search_id_text)
        box_ids = self.cur_shapes_layer.properties.get('bbox_id')
        if box_ids is None:
            show_warning("'bbox_id' property not found in the current shapes layer.")
            return
        try:
            index = list(box_ids).index(search_id)
            self.cur_shapes_layer.selected_data = {index}
            bbox = self.cur_shapes_layer.data[index]
            min_y, min_x = bbox[0, 0], bbox[0, 1]
            max_y, max_x = bbox[2, 0], bbox[2, 1]
            center_y = (min_y + max_y) / 2
            center_x = (min_x + max_x) / 2
            self.viewer.camera.center = (center_y, center_x)
        except ValueError:
            show_warning(f"Detection with ID {search_id} not found in layer {self.cur_shapes_layer.name}.")

    def _setup_timelapse_widget(self):
        """
        Sets up the GUI part for timelapse and tracking.
        """
        main_widget = QWidget()
        main_vbox = QVBoxLayout()

        timelapse_group = QGroupBox('Timelapse selection')
        timelapse_vbox = QVBoxLayout()
        hbox_selector = QHBoxLayout()
        timelapse_label = QLabel('Timelapse: ', self)
        self.timelapse_selection = QComboBox()
        self.timelapse_selection.currentIndexChanged.connect(self._on_timelapse_change)
        hbox_selector.addWidget(timelapse_label, 1)
        hbox_selector.addWidget(self.timelapse_selection, 4)
        timelapse_vbox.addLayout(hbox_selector)

        hbox_buttons = QHBoxLayout()
        export_timelapse_btn = QPushButton("Export Timelapse")
        delete_timelapse_btn = QPushButton("Delete timelapse")
        delete_timelapse_btn.clicked.connect(self._on_delete_timelapse)
        export_timelapse_btn.clicked.connect(self._on_export_timelapse_click)
        hbox_buttons.addWidget(export_timelapse_btn)
        hbox_buttons.addWidget(delete_timelapse_btn)
        timelapse_vbox.addLayout(hbox_buttons)
        timelapse_group.setLayout(timelapse_vbox)
        main_vbox.addWidget(timelapse_group)

        tracking_group = QGroupBox("Tracking")
        tracking_vbox = QVBoxLayout()

        # Tracking method selector
        tracking_method_hbox = QHBoxLayout()
        tracking_method_label = QLabel("Tracking method:", self)
        self.tracking_method_selector = QComboBox()
        self.tracking_method_selector.addItem("TrackPy")
        self.tracking_method_selector.currentIndexChanged.connect(self._on_tracking_method_changed)
        tracking_method_hbox.addWidget(tracking_method_label)
        tracking_method_hbox.addWidget(self.tracking_method_selector)
        tracking_vbox.addLayout(tracking_method_hbox)

        # TrackPy parameters
        self.tracking_params_stack = QStackedLayout()
        trackpy_params_widget = QWidget()
        trackpy_params_layout = QFormLayout()
        self.trackpy_search_range = QLineEdit()
        self.trackpy_search_range.setValidator(QIntValidator(1, 99999))
        trackpy_params_layout.addRow(QLabel("Search Range:"), self.trackpy_search_range)
        self.trackpy_search_range.setText("20")
        self.trackpy_memory = QLineEdit()
        self.trackpy_memory.setValidator(QIntValidator(1, 99999))
        self.trackpy_memory.setText("3")
        trackpy_params_layout.addRow(QLabel("Memory:"), self.trackpy_memory)
        
        self.create_missing_detections_checkbox = QCheckBox("Create missing detections")
        self.create_missing_detections_checkbox.setChecked(False)
        self.create_missing_detections_checkbox.setToolTip("When disabled, only changes detection IDs. If enabled, can create new detections based on memory parameter.")
        trackpy_params_layout.addRow(self.create_missing_detections_checkbox)
        
        trackpy_params_widget.setLayout(trackpy_params_layout)
        self.tracking_params_stack.addWidget(trackpy_params_widget)
        params_container = QWidget()
        params_container.setLayout(self.tracking_params_stack)
        tracking_vbox.addWidget(params_container)

        tracking_btns_hbox = QHBoxLayout()
        self.run_tracking_btn = QPushButton("Run Tracking")
        self.run_tracking_btn.clicked.connect(self._on_run_tracking)
        self.manual_tracking_btn = QPushButton("Manual tracking")
        self.manual_tracking_btn.clicked.connect(self._on_run_manual_tracking)
        tracking_btns_hbox.addWidget(self.run_tracking_btn)
        tracking_btns_hbox.addWidget(self.manual_tracking_btn)
        tracking_vbox.addLayout(tracking_btns_hbox)
        tracking_vbox.addStretch(1)

        tracking_group.setLayout(tracking_vbox)
        main_vbox.addWidget(tracking_group)

        main_widget.setLayout(main_vbox)
        return main_widget

    def _on_tracking_method_changed(self, idx):
        self.tracking_params_stack.setCurrentIndex(idx)

    def _on_run_tracking(self):
        """
        Called when user clicks the run tracking button.
        """
        if not self.cur_timelapse_name:
            show_error("No timelapse selected for tracking.")
            return
        image_layer_name = self.cur_timelapse_name.split('-Labels')[0]
        if image_layer_name not in self.timelapse_image_layers or image_layer_name not in self.image_layer_names:
            show_error(f"Timelapse image '{image_layer_name}' not found.")
            return
        total_frames = self.viewer.layers[image_layer_name].data.shape[0]
        timelapse_shape_names = [f"TL_Frame{frame_idx}_{self.cur_timelapse_name}" for frame_idx in range(total_frames)]
        for shape_layer_name in timelapse_shape_names:
            self.viewer.layers[shape_layer_name].metadata['napari-organoid-analyzer:_tracking'] = True
        if self.tracking_method_selector.currentText() == "TrackPy":
            if not self.trackpy_search_range.text() or not self.trackpy_memory.text():
                show_error("Please specify both search range and memory for TrackPy tracking.")
                return
            trackpy_params = {
                'search_range': int(self.trackpy_search_range.text()),
                'memory': int(self.trackpy_memory.text()),
                'create_missing_detections': self.create_missing_detections_checkbox.isChecked()
            }
            self.organoiDL.run_tracking(self.viewer.layers[image_layer_name].data, timelapse_shape_names, 'trackpy', trackpy_params)
        else:
            show_error("Unsupported tracking method selected.")
            return
        for shape_layer_name in timelapse_shape_names:
            self._update_detections(shape_layer_name)

        for shape_layer_name in timelapse_shape_names:
            del self.viewer.layers[shape_layer_name].metadata['napari-organoid-analyzer:_tracking']
        

    def _on_run_manual_tracking(self):
        pass

    def _on_timelapse_change(self):
        """
        Called when user changes the selected timelapse.
        """
        if self.cur_timelapse_name is not None:
            timelapse_image_name = self.cur_timelapse_name.split('-Labels')[0]
            if timelapse_image_name not in self.timelapse_image_layers or timelapse_image_name not in self.image_layer_names:
                show_error(f"Timelapse image '{timelapse_image_name}' not found.")
                return
            self.viewer.layers[timelapse_image_name].visible = False
            for labels_layer_name in self.timelapses[self.cur_timelapse_name]:
                if labels_layer_name in self.viewer.layers:
                    self.viewer.layers[labels_layer_name].visible = False
            self.viewer.layers.selection.active = None
        self.cur_timelapse_name = self.timelapse_selection.currentText() if len(self.timelapse_selection.currentText()) > 0 else None
        if self.cur_timelapse_name:
            timelapse_image_name = self.cur_timelapse_name.split('-Labels')[0]
            if timelapse_image_name not in self.timelapse_image_layers or timelapse_image_name not in self.image_layer_names:
                show_error(f"Timelapse image '{timelapse_image_name}' not found.")
                return
            self.viewer.layers[timelapse_image_name].visible = True
            self._on_frame_change()

    def _on_delete_timelapse(self):
        """
        Called when user clicks the delete timelapse button.
        """
        if self.cur_timelapse_name is not None:
            if not self.cur_timelapse_name in self.timelapses:
                show_error(f"Timelapse '{self.cur_timelapse_name}' not found.")
                return
            for frame in list(self.timelapses[self.cur_timelapse_name]):
                if frame in self.viewer.layers:
                    self.viewer.layers.remove(frame)
            show_info("Timelapse deleted successfully.")
        else:
            show_warning("No timelapse selected for deletion.")

    def _on_create_labelled_timelapse(self):
        """
        Prompt user to select export file, then for each frame of the timelapse,
        create a screenshot of the timelapse image, bounding box, and, if available,
        segmentation image overlayed together. Merge screenshots into a timelapse and save as mp4.
        """

        if not self.cur_timelapse_name or self.cur_timelapse_name not in self.timelapses:
            show_error("No timelapse selected or timelapse not found.")
            return

        # Prompt user for export file
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Labelled Timelapse", f"{self.cur_timelapse_name}.mp4", "MP4 files (*.mp4)"
        )
        if not file_path:
            show_warning("Export canceled.")
            return

        # Get image layer and frame count
        timelapse_image_name = self.cur_timelapse_name.split('-Labels')[0]
        if timelapse_image_name not in self.viewer.layers:
            show_error(f"Timelapse image layer '{timelapse_image_name}' not found.")
            return
        
        image_layer = self.viewer.layers[timelapse_image_name]
        image_data = image_layer.data
        if image_data.ndim != 4:
            show_error("Selected timelapse image is not a 4D array.")
            return
        total_frames = image_data.shape[0]

        # Prepare screenshots
        screenshots = []
        orig_visibility = {layer.name: layer.visible for layer in self.viewer.layers}
        orig_selection = self.viewer.layers.selection.active
        for layer in self.viewer.layers:
            layer.visible = False
        self.viewer.layers[timelapse_image_name].visible = True
        if timelapse_image_name in self.timelapse_segmentations and self.timelapse_segmentations[timelapse_image_name] in self.viewer.layers:
            self.viewer.layers[self.timelapse_segmentations[timelapse_image_name]].visible = True

        for i in range(total_frames):
            # Show image layer for frame i
            self.viewer.dims.current_step = (i,)

            # Show bbox layer for this frame
            frame_layer_name = f"TL_Frame{i}_{self.cur_timelapse_name}"
            if frame_layer_name in self.viewer.layers:
                self.viewer.layers[frame_layer_name].visible = True
            else:
                show_warning(f"Bounding box layer for frame {i} not found. Skipping overlay for this frame.")

            # Take screenshot
            screenshot = self.viewer.screenshot(canvas_only=True)
            screenshots.append(screenshot)

            # Hide bbox for next frame
            if frame_layer_name in self.viewer.layers:
                self.viewer.layers[frame_layer_name].visible = False

        # Restore original visibility
        for layer in self.viewer.layers:
            if layer.name in orig_visibility:
                layer.visible = orig_visibility[layer.name]
        self.viewer.layers.selection.active = orig_selection

        # Write screenshots to mp4 using cv2
        if not screenshots:
            show_error("No frames to export.")
            return
        height, width = screenshots[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, 2, (width, height))  # 2 fps

        for img in screenshots:
            if img.dtype != np.uint8:
                img = (255 * (img / img.max())).astype(np.uint8)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()

        show_info(f"Labelled timelapse exported to {file_path}")

    def _on_frame_change(self):
        """
        Called when user changes the selected frame in the timelapse.
        """
        if self.cur_timelapse_name is not None:
            cur_frame = self.viewer.dims.current_step[0]
            # Hide all timelapse layers
            for layer_name in self.timelapses[self.cur_timelapse_name]:
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].visible = False
            self.viewer.layers.selection.active = None
            # Show only the layer corresponding to the current frame, if it exists
            frame_layer_name = f"TL_Frame{cur_frame}_{self.cur_timelapse_name}"
            if frame_layer_name in self.viewer.layers:
                self.viewer.layers[frame_layer_name].visible = True
                self.viewer.layers.selection.active = self.viewer.layers[frame_layer_name]

    def _get_layer_names(self, layer_type: layers.Layer = layers.Image) -> List[str]:
        """
        Get a list of layer names of a given layer type.
        """
        layer_names = [layer.name for layer in self.viewer.layers if type(layer) == layer_type]
        return layer_names
    
    def _on_labels_layer_change(self):
        """
        Called when user changes layer of labels used for segmentation
        """
        self.label_layer_name = self.segmentation_image_layer_selection.currentText()
        # Show or hide the "Run for entire timelapse" checkbox based on layer name
        if self.label_layer_name.startswith("TL_Frame"):
            self.run_for_timelapse_checkbox.setVisible(True)
        else:
            self.run_for_timelapse_checkbox.setVisible(False)    
        self.signals_list.setRowCount(0)
        if self.label_layer_name in self.label2im and self.label2im[self.label_layer_name] in self.im2signal:
            image_name = self.label2im[self.label_layer_name]
            self.signals_list.setRowCount(len(self.im2signal[image_name].items()))
            for row, (signal_name, layer_name) in enumerate(self.im2signal[image_name].items()):

                signal_item = QTableWidgetItem(signal_name)
                self.signals_list.setItem(row, 0, signal_item)

                layer_item = QTableWidgetItem(layer_name)
                self.signals_list.setItem(row, 1, layer_item)

                delete_btn = QPushButton()
                delete_btn.setText("Delete")
                delete_btn.clicked.connect(lambda: self._delete_signal_matching(image_name, signal_name))
                btn_container = QWidget()
                btn_layout = QHBoxLayout(btn_container)
                btn_layout.addWidget(delete_btn)
                btn_layout.setAlignment(Qt.AlignCenter)
                self.signals_list.setCellWidget(row, 2, btn_container)

    def _delete_signal_matching(self, image_name, signal_name):
        self.im2signal[image_name].pop(signal_name, None)
        self._on_labels_layer_change()

    def _on_layer_name_change(self, event):
        """
        Called whether user changes the name of any of the layers.
        """
        
        # Update selectors for image and shapes layers
        self.segmentation_image_layer_selection.clear()
        self.annotation_image_layer_selection.clear()
        for name in self._get_layer_names(layer_type=layers.Shapes): 
            self.segmentation_image_layer_selection.addItem(name)
            self.annotation_image_layer_selection.addItem(name)

        self.image_layer_selection.clear()
        for name in self._get_layer_names(layer_type=layers.Image):
            self.image_layer_selection.addItem(name)

        # TODO: Handle layer name change 

    def _on_shape_selected(self, event):
        """
        Called when user changes the selection of a shape in layer
        """
        if self.cur_shapes_layer is not None and len(self.cur_shapes_layer.selected_data) != 0:
            self._update_detection_data_tab()
        
    def _update_detection_data_tab(self):
        """
        Updates the "Detection data" tab based on the current shapes layer and selected data.
        """
        self.detection_data_tree.clear()  # Clear previous data
        if self.cur_shapes_layer and self.cur_shapes_layer.selected_data:
            self.tab_widget.setTabEnabled(1, True)
            for index in self.cur_shapes_layer.selected_data:
                # Create a top-level item for each selected shape
                top_item = QTreeWidgetItem(self.detection_data_tree)
                top_item.setText(0, f"Detection ID {self.cur_shapes_layer.properties['bbox_id'][index]}")
                top_item.setExpanded(False)

                # Add properties as child items
                for prop_name, prop_values in self.cur_shapes_layer.properties.items():
                    if prop_name != 'bbox_id':
                        child_item = QTreeWidgetItem(top_item)
                        child_item.setText(0, prop_name)
                        child_item.setText(1, str(prop_values[index]))
            self.detection_data_tree.expandAll()

    def _export_detection_data_to_csv(self):
        """
        Export the detection data displayed in the "Detection data" tab to a CSV file.
        """
        if not self.cur_shapes_layer or not self.cur_shapes_layer.selected_data:
            show_warning("No detection data to export.")
            return
        data = []
        for index in self.cur_shapes_layer.selected_data:
            row = {}
            for prop_name, prop_values in self.cur_shapes_layer.properties.items():
                    row[prop_name] = prop_values[index]
            data.append(row)

        # Convert to pandas DataFrame
        df = pd.DataFrame(data)

        # Open a dialog to select the file to save
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Detection Data", "detection_data.csv", "CSV files (*.csv)")
        if file_path:
            df.to_csv(file_path, index=False)
            show_info(f"Detection data exported successfully to {file_path}.")
        else:
            show_warning("Export canceled.")

    def _setup_labels_for_annotation_widget(self):
        """
        Sets up the GUI part for selecting which labels to use for annotation.

        Contains a 
        - select labels layer dropdown
        - apply to whole timelapse checkbox
        - apply on tracked organoids checkbox
        """
        widget = QGroupBox('Select labels layer')
        vbox = QVBoxLayout()

        # Select labels layer
        hbox_config = QHBoxLayout()
        hbox_config.addWidget(QLabel("Labels Layer:"), 1)
        self.annotation_image_layer_selection = QComboBox()
        hbox_config.addWidget(self.annotation_image_layer_selection, 4)
        vbox.addLayout(hbox_config)
        
        # Run for entire timelapse checkbox
        self.annotation_run_for_timelapse_checkbox = QCheckBox("Run for entire timelapse")
        self.annotation_run_for_timelapse_checkbox.setVisible(False)
        self.annotation_run_for_timelapse_checkbox.setChecked(False)
        vbox.addWidget(self.annotation_run_for_timelapse_checkbox)
        
        widget.setLayout(vbox)
        return widget
        
    def _setup_create_annotation_feature_widget(self):
        """
        Sets up the GUI part for configuring the annotation feature.

        Contains a 
        - select annotation type dropdown: Should support Text, Ruler, Objects, 
            Number, Classes
        - feature name text field: This feature name must be unique and not 
            exist in the data or in the created features already.
        - start annotating button
        """
        widget = QGroupBox('Create annotation')
        vbox = QVBoxLayout()
        
        # Feature name
        hbox_config1 = QHBoxLayout()
        feature_name_desc = QLabel('Feature name: ', self)
        self.new_feature_name = QLineEdit()
        hbox_config1.addWidget(feature_name_desc, 1)
        hbox_config1.addWidget(self.new_feature_name, 4)
        vbox.addLayout(hbox_config1)

        # Feature config
        hbox_config2 = QHBoxLayout()
        feature_type_desc = QLabel('Type: ', self)
        feature_type_desc.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.new_feature_type_selection = QComboBox()
        for ft in ANNOTATION_TYPES:
            self.new_feature_type_selection.addItem(ft)

        start_annotation = QPushButton('Start annotating')
        start_annotation.clicked.connect(self._on_create_annotation_feature)
        start_annotation.setStyleSheet("border: 0px")

        hbox_config2.addWidget(feature_type_desc, 1)
        hbox_config2.addWidget(self.new_feature_type_selection, 3)
        hbox_config2.addWidget(start_annotation, 1)
        hbox_config2.addSpacing(15)
        hbox_config2.addStretch(1)   
        vbox.addLayout(hbox_config2)     
        widget.setLayout(vbox)
        return widget

    def _setup_continue_annotation_widget(self):
        """
        Sets up the GUI part for continuing previously defined annotation.

        Contains a select feature dropdown, delete feature, and continue annotating button.
        """
        widget = QGroupBox('Resume annotation')
        vbox = QVBoxLayout()
        
        # Feature name
        hbox_config1 = QHBoxLayout()
        hbox_config1.addWidget(QLabel('Annotation name:'), 1)
        self.resume_feature_name = QComboBox()
        for ft in self._load_annotation_features():
            self.resume_feature_name.addItem(ft)
        hbox_config1.addWidget(self.resume_feature_name, 4)
        vbox.addLayout(hbox_config1)

        # Feature action
        hbox_config2 = QHBoxLayout()
        delete_feature = QPushButton('Delete')
        delete_feature.clicked.connect(self._on_delete_annotation_feature)
        delete_feature.setStyleSheet("border: 0px")

        start_annotation = QPushButton('Continue annotating')
        start_annotation.clicked.connect(self._on_continue_annotation)
        start_annotation.setStyleSheet("border: 0px")

        hbox_config2.addWidget(delete_feature)
        hbox_config2.addWidget(start_annotation)
        vbox.addLayout(hbox_config2) 

        widget.setLayout(vbox)
        return widget

    def _on_create_annotation_feature(self):
        annotation_name = f"{self.new_feature_name.text()}_{self.annotation_image_layer_selection.currentText()}"
        feature = {
            annotation_name: {
                'property_name': self.new_feature_name.text(),
                'type': self.new_feature_type_selection.currentText(),
            }
        }
        annotation_features = session.SESSION_VARS.get('annotation_features', {})
        if annotation_name in annotation_features:
            raise ValueError(f"Annotation of name {annotation_name} already exists. Please choose a different name or delete existing annotation")
        
        # Add annotation feature to cached setting
        annotation_features.update(feature)
        session.set_session_var('annotation_features', annotation_features)

        # Add annotation feature to resume annotation dropdown
        self.resume_feature_name.addItem(annotation_name)

        self.annotate(feature)

    def _load_annotation_features(self):
        annotation_features = session.SESSION_VARS.get('annotation_features', {})

        # Return list of existing features
        return list(annotation_features.keys())
    
    def _on_delete_annotation_feature(self):
        feature_name = self.resume_feature_name.currentText()
        annotation_features = session.SESSION_VARS.get('annotation_features', {})

        # Delete feature from cached settings
        annotation_features.pop(feature_name, None)
        session.set_session_var('annotation_features', annotation_features)

        # Delete feature from the resume annotation dropdown
        item_id = self.resume_feature_name.findText(feature_name)
        self.resume_feature_name.removeItem(item_id)
    
    def _on_continue_annotation(self):
        feature_name = self.resume_feature_name.currentText()
        annotation_features = session.SESSION_VARS.get('annotation_features', {})
        feature = annotation_features.get(feature_name, None)
        if feature is None:
            raise RuntimeError(f"Feature with name {feature_name} was not found.")
        if not feature['type'] in ANNOTATION_TYPES:
            raise RuntimeError(f"Feature with name {feature_name} has unknown annotation type {feature['type']}")
        self.annotate({feature_name: feature})

    def annotate(self, feature):
        """Starts the annotation loop with custom annotation widgets depending on the feature type."""
        annotation_name = next(iter(feature))
        annotation_data = feature[annotation_name]
        annotation_data.update({"annotation_name": annotation_name})
        
        label_layer_name = self.annotation_image_layer_selection.currentText()
        if not label_layer_name:
            raise ValueError("No label layer selected for annotation. Please select a label layer and try again.")
        if label_layer_name not in self.viewer.layers or label_layer_name not in self.label2im:
            raise ValueError(f"Label layer {label_layer_name} not found in viewer or doesn't have corresponding image.")
        labels_layer = self.viewer.layers[label_layer_name]
        
        image_layer_name = self.label2im[label_layer_name]
        if image_layer_name not in self.viewer.layers:
            raise ValueError(f"Image layer {image_layer_name} not found in viewer")
        
        if label_layer_name.startswith("TL_Frame"):
            frame_idx = int(self.label_layer_name.split('_')[1][5:])
            image = self.viewer.layers[image_layer_name].data[frame_idx]
        else:
            image = self.viewer.layers[image_layer_name].data
        bboxes = labels_layer.data
        bboxes = np.array(convert_boxes_from_napari_view(bboxes))
        properties = labels_layer.properties.copy()

        # for property_name, property in properties.items():
        #     if len(property) != bboxes.shape[0]:
        #         raise RuntimeError(f"Number of properties for property save_annotation {property_name} ({len(property)}) doesn't match number of bounding boxes ({bboxes.shape[0]})")
            
        annotation_dialogue = get_annotation_dialogue(image, bboxes, properties, annotation_data, self)
        if annotation_dialogue.exec() != QDialog.Accepted:
            show_warning("Annotation cancelled. But your changes have been saved.")
            return
        new_annotations = annotation_dialogue.get_annotations()
        if annotation_data['type'] == "Ruler":
            pass
            # TODO: update features from within the annotation process
            # property_names = [f"{annotation_data['property_name']}_line", 
            #                   f"{annotation_data['property_name']}_total_length",
            #                   f"{annotation_data['property_name']}_average_length",
            #                   f"{annotation_data['property_name']}_count"
            #                   ]
            # for idx, property_name in enumerate(property_names):
            #     if property_name in properties:
            #         feature_data = properties[property_name]
            #     else:
            #         feature_data = ["" for i in range(len(properties['bbox_id']))]
            #     cur_box_ids = properties['bbox_id']
            #     for box_id, value in new_annotations.items():
            #         arr_id = np.where(cur_box_ids == int(box_id))[0][0]
            #         feature_data[arr_id] = value[idx]
            #         properties.update({property_name: feature_data})
        else:
            if annotation_data['property_name'] in properties:
                feature_data = properties[annotation_data['property_name']]
            else:
                feature_data = ["" for i in range(len(properties['bbox_id']))]
            cur_box_ids = properties['bbox_id']
            for box_id, value in new_annotations.items():
                arr_id = np.where(cur_box_ids == int(box_id))[0][0]
                feature_data[arr_id] = value
                properties.update({annotation_data['property_name']: feature_data})
        labels_layer.properties = properties
        self._update_detection_data_tab()

    def _on_add_signal(self):
        signal_dialog = SignalDialog(self, self._get_layer_names())

        if signal_dialog.exec() != QDialog.Accepted:
            show_warning("Signal import cancelled")
            return
        
        image_name = signal_dialog.get_target()
        from_existing, signal_uri = signal_dialog.get_source()
        signal_name = signal_dialog.get_name()

        if not from_existing:
            old_layers = self._get_layer_names()
            self.viewer.open(signal_uri)
            curr_layers = self._get_layer_names()
            new_layers = [name for name in curr_layers if not name in old_layers]
            if len(new_layers) != 1:
                raise RuntimeError(f"Error importing signal. New layers encountered {new_layers} (1 expected)")
            signal_uri = new_layers[0]
        image_shape = self.viewer.layers[image_name].data.shape
        signal_shape = self.viewer.layers[signal_uri].data.shape
        if not (
            (len(signal_shape) == len(image_shape) - 1 and signal_shape == image_shape[:-1]) or 
            (len(signal_shape) == len(image_shape) and signal_shape[:-1] == image_shape[:-1]) or
            (len(signal_shape) == 3 and len(image_shape) == 2 and signal_shape[:-1] == image_shape)
        ):
            raise RuntimeError(f"Signal dimensions of {signal_shape} do not correspond to image dimensions {image_shape}. Signal not added")
        if not image_name in self.im2signal:
            self.im2signal[image_name] = {signal_name: signal_uri}
        else:
            self.im2signal[image_name].update({signal_name: signal_uri})
        self.im2signal[image_name].update({signal_name: signal_uri})
        self._on_labels_layer_change()
