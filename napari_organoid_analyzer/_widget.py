from pathlib import Path
from typing import List

from skimage.io import imsave
from skimage.color import rgb2gray
from datetime import datetime
import json
import os.path
import pandas as pd
from napari.utils import progress

from napari import layers
from napari.utils.notifications import show_info, show_error, show_warning
from urllib.request import urlretrieve
from napari_organoid_analyzer._utils import (
    convert_boxes_from_napari_view, 
    collate_instance_masks, 
    compute_image_hash, 
    convert_boxes_to_napari_view,
    validate_bboxes,
    get_timelapse_name
)
from napari_organoid_analyzer._widgets.annotation import get_annotation_dialogue


import numpy as np

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHeaderView, 
    QAbstractItemView,
    QTableWidgetItem,
    QWidget, 
    QVBoxLayout, 
    QApplication, 
    QDialog, 
    QFileDialog, 
    QGroupBox, 
    QHBoxLayout, 
    QLabel, 
    QComboBox, 
    QPushButton, 
    QLineEdit, 
    QProgressBar, 
    QSlider, 
    QTabWidget, 
    QTreeWidget, 
    QTreeWidgetItem, 
    QCheckBox, 
    QTableWidget
)

from napari_organoid_analyzer._orgacount import OrganoiDL
from napari_organoid_analyzer import _utils as utils
from napari_organoid_analyzer import settings
from napari_organoid_analyzer import session
from napari_organoid_analyzer._widgets.dialogues import ConfirmUpload, ConfirmSamUpload, ExportDialog, SignalDialog, SignalChannelDialog
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

import warnings
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

        # Add tabs to the tab widget
        self.tab_widget.addTab(self.configuration_tab, "Configuration")
        self.tab_widget.addTab(self.detection_data_tab, "Detection data")
        self.tab_widget.addTab(self.annotation_tab, "Add Annotation")
        self.tab_widget.setTabEnabled(1, False)  # Initially disable the "Detection data" tab
        self.tab_widget.setTabEnabled(2, True)  # Always enable the "Add Annotation" tab

        # Set up the layout for the configuration tab
        self.configuration_tab.setLayout(QVBoxLayout())
        self.configuration_tab.layout().addWidget(self._setup_input_widget())
        self.configuration_tab.layout().addWidget(self._setup_output_widget())
        self.configuration_tab.layout().addWidget(self._setup_segmentation_widget())
        self.configuration_tab.layout().addWidget(self._setup_timelapse_widget())

        # Set up the layout for the detection data tab
        self.detection_data_tab.setLayout(QVBoxLayout())
        self.detection_data_tree = QTreeWidget()
        self.detection_data_tree.setHeaderLabels(["Detections", "Properties"])
        self.detection_data_tab.layout().addWidget(self.detection_data_tree)

        # Set up the layout for the add annotation tab
        self.annotation_tab.setLayout(QVBoxLayout())
        self.annotation_tab.layout().addWidget(self._setup_labels_for_annotation_widget())
        self.annotation_tab.layout().addWidget(self._setup_create_annotation_feature_widget())
        self.annotation_tab.layout().addWidget(self._setup_continue_annotation_widget())

        # Add export button below the tree view
        export_button = QPushButton("Export Selected")
        export_button.clicked.connect(self._export_detection_data_to_csv)
        self.detection_data_tab.layout().addWidget(export_button)

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
        
        with open(cache_file, 'w') as f:
            bboxes = self.organoiDL.pred_bboxes[layer_name]
            box_ids = self.organoiDL.pred_ids[layer_name]
            scores = self.organoiDL.pred_scores[layer_name]
            scale = self.viewer.layers[corr_image_name].scale[:2]

            # Create a dictionary to store the data
            cache_data = {
                'bboxes': bboxes.tolist(),
                'bbox_ids': list(map(int, box_ids)),
                'scores': scores.tolist(),
                'scale': scale.tolist(),
            }
                
            # Write the data to the cache file
            json.dump(cache_data, f)
                
        self.cache_index[image_hash] = cache_file
        self._save_cache_index()

    
    def _load_cached_results(self, cache_file):
        """Load detection results from cache file"""
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                return cache_data
        except (json.JSONDecodeError, IOError):
            show_error(f"Failed to load cached results from {cache_file}")
            return None
        
    
    def _create_shapes_from_cache(self, image_layer_name, cache_data, labels_layer_name=None):
        """Create a shapes layer from cached detection data"""
        if self.organoiDL.img_scale[0] == 0:
            self.organoiDL.set_scale(self.viewer.layers[image_layer_name].scale[:2])
            
        bboxes = convert_boxes_to_napari_view(np.array(cache_data.get('bboxes', [])))
        box_ids = list(map(int, cache_data.get('bbox_ids', [])))
        scores = cache_data.get('scores', [])
        labels = cache_data.get('labels', [0] * len(bboxes))
        scale = cache_data.get('scale', self.viewer.layers[image_layer_name].scale[:2])

        if scale[0] != self.viewer.layers[image_layer_name].scale[0] or scale[1] != self.viewer.layers[image_layer_name].scale[1]:
            show_warning("Scale mismatch between cached data and current image layer")

        if len(bboxes) == 0:
            show_error("No detections found in cache")
            return False
            
        # Create a new shapes layer
        if labels_layer_name is None:
            labels_layer_name = f'{image_layer_name}-Labels-Cache-{datetime.strftime(datetime.now(), "%H_%M_%S")}'

        self.organoiDL.update_bboxes_scores(labels_layer_name, bboxes, scores, box_ids, 0, 0)
        bboxes, scores, box_ids = self.organoiDL.apply_params(labels_layer_name, self.confidence, self.min_diameter)
        

        # Set up the shapes layer
        properties = {'box_id': box_ids, 'confidence': scores}
        text_params = {'string': 'ID: {box_id}\nConf.: {confidence:.2f}',
                       'size': 12,
                       'anchor': 'upper_left',
                       'color': settings.TEXT_COLOR}
        
        self.cur_shapes_layer = self.viewer.add_shapes(
            bboxes, 
            name=labels_layer_name,
            scale=scale,
            face_color='transparent',
            properties=properties,
            text=text_params,
            edge_color=settings.COLOR_DEFAULT,
            shape_type='rectangle',
            edge_width=12
        )
        
        self.label2im[labels_layer_name] = image_layer_name
        self.stored_confidences[labels_layer_name] = self.confidence
        self.stored_diameters[labels_layer_name] = self.min_diameter
        self._update_cur_shapes_layer_label_and_checkbox()
        self._update_num_organoids(len(bboxes))

        if labels_layer_name.startswith("TL_Frame"):
            timelapse_name = get_timelapse_name(labels_layer_name)
            if not timelapse_name in self.timelapses:
                self.timelapses[timelapse_name] = set()
                self.timelapse_selection.addItem(timelapse_name)
                self.timelapse_selection.setCurrentText(timelapse_name)
                self.cur_timelapse_name = timelapse_name
            self.timelapses[timelapse_name].add(labels_layer_name)
        
        self.cur_shapes_layer.events.data.connect(self.shapes_event_handler)
        
        # Most likely action after loading cached detections is to select/delete/change some of them.
        self.viewer.layers[labels_layer_name].mode = 'select'
        
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
        if self.cur_shapes_layer and self.cur_shapes_layer.name.startswith("TL_Frame"):
            self.apply_to_timelapse_checkbox.setVisible(True)
        else:
            self.apply_to_timelapse_checkbox.setVisible(False)

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
            from qtpy.QtWidgets import QMessageBox, QCheckBox, QVBoxLayout
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

    def _update_detections(self, bboxes, scores, box_ids, labels_layer_name):
        """ Adds the shapes layer to the viewer or updates it if already there """
        self._update_num_organoids(len(bboxes))
        # if layer already exists
        if labels_layer_name in self.shape_layer_names:
            self.viewer.layers[labels_layer_name].data = bboxes # hack to get edge_width stay the same!
            # IMPORTANT!!! Assignment of properties is possible only in its entirety. Example code below would not work
            # self.viewer.layers[labels_layer_name].properties['box_id'] = box_ids
            # self.viewer.layers[labels_layer_name].properties['confidence'] = scores
            self.viewer.layers[labels_layer_name].properties = { 'box_id': box_ids, 
                                                                'confidence': scores}
            self.viewer.layers[labels_layer_name].edge_width = 12
            self.viewer.layers[labels_layer_name].refresh()
            self.viewer.layers[labels_layer_name].refresh_text()
            self.cur_shapes_layer = self.viewer.layers[labels_layer_name]
        # or if this is the first run
        else:
            text_params = {'string': 'ID: {box_id}\nConf.: {confidence:.2f}',
                            'size': 12,
                            'anchor': 'upper_left',
                            'color': settings.TEXT_COLOR}
            # if no organoids were found just make an empty shapes layer
            if self.num_organoids==0: 
                self.cur_shapes_layer = self.viewer.add_shapes(name=labels_layer_name,
                                                               properties={'box_id': [],'confidence': []},
                                                               text=text_params,
                                                               edge_color=settings.COLOR_DEFAULT,
                                                               face_color='transparent',
                                                               edge_width=12,
                                                               scale=self.viewer.layers[self.image_layer_name].scale[:2],)
            # otherwise make the layer and add the boxes
            else:
                properties = {'box_id': box_ids,'confidence': scores}
                self.cur_shapes_layer = self.viewer.add_shapes(bboxes, 
                                                               name=labels_layer_name,
                                                               scale=self.viewer.layers[self.image_layer_name].scale[:2],
                                                               face_color='transparent',  
                                                               properties = properties,
                                                               text = text_params,
                                                               edge_color=settings.COLOR_DEFAULT,
                                                               shape_type='rectangle',
                                                               edge_width=12) # warning generated here
                            
        # set current_edge_width so edge width is the same when users annotate - doesnt' fix new preds being added!
        self.viewer.layers[labels_layer_name].current_edge_width = 1
        self.viewer.layers[labels_layer_name].mode = 'select'
        self.stored_confidences[labels_layer_name] = self.confidence
        self.stored_diameters[labels_layer_name] = self.min_diameter
        self._update_cur_shapes_layer_label_and_checkbox()

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
            self.label2im[labels_layer_name] = self.image_layer_name
            self.viewer.window._status_bar._toggle_activity_dock(True)
            self._detect_organoids(img_data, labels_layer_name)
        elif img_data.ndim == 4:
            if not self.guidance_layer_name is None and not validate_bboxes(self.viewer.layers[self.guidance_layer_name].data, img_data.shape[1:3]):
                show_error(f"Bboxes from guidance layer {self.guidance_layer_name} cannot be applied to image {self.image_layer_name} with shape {img_data.shape[:2]}")
                return
            timelapse_name = f'{self.image_layer_name}-Labels-{self.model_name}-{datetime.strftime(datetime.now(), "%H_%M_%S")}'
            frame_names = []
            self.viewer.window._status_bar._toggle_activity_dock(True)
            for i in progress(range(img_data.shape[0])):
                labels_layer_name = f'TL_Frame{i}_{timelapse_name}'
                self.label2im[labels_layer_name] = self.image_layer_name
                self._detect_organoids(img_data[i], labels_layer_name)
                frame_names.append(labels_layer_name)
            self.cur_timelapse_name = timelapse_name
            self.timelapses[timelapse_name] = frame_names
            self.timelapse_selection.addItem(timelapse_name)
            self.timelapse_selection.setCurrentText(timelapse_name)
        else:
            show_error(f"Wrong format for image with shapes {img_data.ndim}")
            
        self.viewer.window._status_bar._toggle_activity_dock(False)
            
        
        # check if the image is not grayscale and convert it

    def _detect_organoids(self, img_data, labels_layer_name):
        """
        Detect organoids from the image (or timelapse frame) and create a shapes layer
        """

        loaded_cached_data = self.compute_and_check_image_hash(img_data, self.label2im[labels_layer_name])
        if loaded_cached_data:
            return

        if img_data.ndim == 3:
            if img_data.shape[2] == 4:
                img_data = img_data[:, :, :3]
            img_data = rgb2gray(img_data)
            img_data = (img_data * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        
        # update the viewer with the new bboxes
        self.stored_confidences[labels_layer_name] = self.confidence
        self.stored_diameters[labels_layer_name] = self.min_diameter

        if labels_layer_name in self.shape_layer_names:
            show_info('Found existing labels layer. Please remove or rename it and try again!')
            return 
        
        crops = convert_boxes_from_napari_view(self.viewer.layers[self.guidance_layer_name].data).tolist() if not self.guidance_layer_name is None else [[0, 0, img_data.shape[0], img_data.shape[1]]]

        # run inference
        self.organoiDL.run(img_data, 
                           labels_layer_name,
                           self.window_sizes,
                           self.downsampling,
                           self.window_overlap,
                           crops)
        
        # set the confidence threshold, remove small organoids and get bboxes in format to visualize
        bboxes, scores, box_ids = self.organoiDL.apply_params(labels_layer_name, self.confidence, self.min_diameter)
        
        # update widget with results
        self._update_detections(bboxes, scores, box_ids, labels_layer_name)
        self._save_cache_results(labels_layer_name)
        # and update cur_shapes_name to newly created shapes layer

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
                masks, features, signal_masks = self.organoiDL.run_segmentation(frame, frame_layer_name, bboxes, frame_signal)
                final_image[i] = collate_instance_masks(masks, color=True)
                for signal_name, signal_seg in signal_masks.items():
                    final_signal_seg[signal_name][i] = collate_instance_masks(signal_seg, color=False)
                if len(labels_layer.properties['box_id']) != masks.shape[0] or len(labels_layer.properties['confidence']) != masks.shape[0]:
                    show_error(f"Mismatch in number of masks and labels for layer {frame_layer_name}. Features have not been updated")
                    continue
                tmp_dict = labels_layer.properties.copy()
                tmp_dict.update(features)
                labels_layer.properties = tmp_dict


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
            masks, features, signal_masks = self.organoiDL.run_segmentation(image_data, self.label_layer_name, bboxes, merged_signal_data)
    
            self.viewer.add_image(collate_instance_masks(masks, color=True), name=segmentation_layer_name, blending='additive')
            for signal_name, signal_mask in signal_masks.items():
                signal_seg_layer_name = f"Segmentation-{signal_name}-{self.label_layer_name}-{datetime.strftime(datetime.now(), '%H_%M_%S')}"
                self.viewer.add_image(collate_instance_masks(signal_mask, color=False), name=signal_seg_layer_name, blending='additive', colormap="red")
            if len(labels_layer.properties['box_id']) != masks.shape[0] or len(labels_layer.properties['confidence']) != masks.shape[0]:
                show_error("Mismatch in number of masks and labels. Please rerun the segmentation.")
                return
            tmp_dict = labels_layer.properties.copy()
            tmp_dict.update(features)
            labels_layer.properties = tmp_dict
    
        self._update_detection_data_tab()
        self.viewer.window._status_bar._toggle_activity_dock(False)
        show_info("Segmentation completed and added to the viewer.")

    def _on_export_click(self):
        """
        Runs when the Export button is clicked to open the export dialog
        and handle the user's selections.
        """
        if not self.label_layer_name:
            show_error("No label layer selected. Please select a label layer and try again.")
            return
        
        label_layer = self.viewer.layers[self.label_layer_name]
        if label_layer is None:
            show_error(f"Layer '{self.label_layer_name}' not found in the viewer.")
            return
        
        lengths = [len(v) for v in label_layer.properties.values()]
        if len(set(lengths)) != 1:
            show_error("Mismatch in number of masks and labels. Please rerun the segmentation on selected layer.")
            return
        
        # Get available features from the layer properties
        available_features = []
        if hasattr(label_layer, 'properties') and label_layer.properties:
            available_features = [k for k in label_layer.properties.keys()]

        masks_available = (self.label_layer_name in self.organoiDL.pred_masks)
        
        # Open the export dialog
        export_dialog = ExportDialog(self, available_features, masks_available)
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
        if export_options['bboxes']:
            self._export_bboxes(label_layer, export_path)
            exported_items.append("bounding boxes")
        
        if export_options['instance_masks']:
            self._export_instance_masks(label_layer, export_path)
            exported_items.append("instance masks")

        if export_options['collated_mask']:
            self._export_collated_masks(label_layer, export_path)
            exported_items.append("collated mask")
        
        if export_options['features']:
            self._export_features(label_layer, export_path, selected_features)
            exported_items.append("features")
        
        if exported_items:
            show_info(f"Export completed successfully to {str(export_path)}\nExported: {', '.join(exported_items)}")
        else:
            show_warning("No items were selected for export.")

    def _export_bboxes(self, label_layer, export_path: Path):
        """Export bounding boxes to JSON file"""
        if self.run_for_timelapse_checkbox.isVisible() and self.run_for_timelapse_checkbox.isChecked():
            
            if not label_layer.name.startswith("TL_Frame"):
                raise RuntimeError("Internal error: Timelapse checkbox is checked but current layer is not a timelapse frame.")
            
            timelapse_name = get_timelapse_name(label_layer.name)

            if timelapse_name not in self.timelapses:
                raise RuntimeError(f"Timelapse '{timelapse_name}' not found.")
            
            timelapse_layers = self.timelapses[timelapse_name]
            data_json = {}
            for label_layer_name in timelapse_layers:
                if not label_layer_name in self.viewer.layers:
                    raise RuntimeError(f"Label layer {label_layer_name} not found in viewer")
                if not label_layer_name.startswith("TL_Frame"):
                    raise RuntimeError(f"Layer {label_layer_name} is in timelapse but not a timelapse frame")
                frame_layer = self.viewer.layers[label_layer_name]
                if len(frame_layer.data) == 0:
                    show_warning(f"No bboxes detected in layer {label_layer_name}. Skippingl...")
                    continue
                frame_idx = int(label_layer_name.split('_')[1][5:])
                frame_data = utils.get_bboxes_as_dict(
                    frame_layer.data,
                    frame_layer.properties['box_id'],
                    frame_layer.properties['confidence'],
                    frame_layer.scale,
                )
                data_json.update({frame_idx: frame_data})            
        else:
            bboxes = label_layer.data
        
            if len(bboxes) == 0: 
                show_warning('No organoids detected! Skipping bounding box export.')
                return
        
            data_json = utils.get_bboxes_as_dict(
                bboxes, 
                label_layer.properties['box_id'],
                label_layer.properties['confidence'],
                label_layer.scale,
            )
            
        # Write bbox coordinates to json
        json_file_path = export_path / f"{self.label_layer_name}_bboxes.json"
        utils.write_to_json(json_file_path, data_json)

    def _export_instance_masks(self, label_layer, export_path: Path):
        """Export instance masks to NPY"""

        if self.run_for_timelapse_checkbox.isVisible() and self.run_for_timelapse_checkbox.isChecked():
            
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
                if not label_layer_name in self.organoiDL.pred_masks:
                    show_warning(f"No masks found for layer {label_layer_name}. Skippingl...")
                    continue
                instance_masks = self.organoiDL.pred_masks[label_layer_name]
                frame_idx = int(label_layer_name.split('_')[1][5:])
                box_ids = self.viewer.layers[label_layer_name].properties['box_id']
                mask_dict = {int(box_ids[i]): instance_masks[i] for i in range(len(instance_masks))}
                file_path = export_folder / f"Frame_{frame_idx}"
                np.save(file_path, mask_dict)
        else:
        
            instance_masks = self.organoiDL.pred_masks[self.label_layer_name]
            if len(instance_masks) == 0:
                show_warning("No masks found for segmentation. Skipping mask export.")
                return
        
            # Export instance masks
            box_ids = label_layer.properties['box_id']
            mask_dict = {int(box_ids[i]): instance_masks[i] for i in range(len(instance_masks))}
            
            instance_mask_file_path = export_path / f"{self.label_layer_name}_instance_masks.npy"
            np.save(instance_mask_file_path, mask_dict)

    def _export_collated_masks(self, label_layer, export_path: Path):
        """Export collated mask to NPY"""
        if self.run_for_timelapse_checkbox.isVisible() and self.run_for_timelapse_checkbox.isChecked():
            
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
                if not label_layer_name in self.organoiDL.pred_masks:
                    show_warning(f"No masks found for layer {label_layer_name}. Skippingl...")
                    continue
                instance_masks = self.organoiDL.pred_masks[label_layer_name]
                frame_idx = int(label_layer_name.split('_')[1][5:])
                collated_mask = collate_instance_masks(instance_masks)
                file_path = export_folder / f"Frame_{frame_idx}"
                np.save(file_path, collated_mask)
        else: 
        
            instance_masks = self.organoiDL.pred_masks[self.label_layer_name]
            if len(instance_masks) == 0:
                show_warning("No masks found for segmentation. Skipping mask export.")
                return

            collated_mask = collate_instance_masks(instance_masks)
            collated_mask_file_path = export_path / f"{self.label_layer_name}_collated_mask.npy"
            np.save(collated_mask_file_path, collated_mask)

    def _export_features(self, label_layer, export_path: Path, selected_features):
        """Export selected features to CSV"""
        # Extract only the selected features
        features_to_export = {}
        if self.run_for_timelapse_checkbox.isVisible() and self.run_for_timelapse_checkbox.isChecked():
            
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
                    if not feature in frame_layer.properties and feature != "Bounding box":
                        show_warning(f"Feature {feature} not found for layer {label_layer_name}. Skipping...")
                        continue
                    if feature == "Bounding box":
                        features_to_export[feature].extend(convert_boxes_from_napari_view(frame_layer.data).tolist())
                        feature_sizes[feature] = len(frame_layer.data)
                    else:
                        features_to_export[feature].extend(frame_layer.properties[feature])
                        feature_sizes[feature] = len(frame_layer.properties[feature])
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
                elif feature == "Bounding box":
                    features_to_export[feature] = convert_boxes_from_napari_view(label_layer.data).tolist()
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
        import shutil
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
        if self.organoiDL.img_scale[0]==0: self.organoiDL.set_scale(self.cur_shapes_layer.scale)

        # make sure to add info to cur_shapes_layer.metadata to differentiate this action from when user adds/removes boxes
        with utils.set_dict_key( self.cur_shapes_layer.metadata, 'napari-organoid-counter:_rerun', True):
            # first update bboxes in organoiDLin case user has added/removed
            if self.apply_to_timelapse_checkbox.isVisible() and self.apply_to_timelapse_checkbox.isChecked():
                if not self.cur_shapes_layer.name.startswith("TL_Frame"):
                    raise RuntimeError("Internal error: Timelapse checkbox is checked but current layer is not a timelapse frame.")
                timelapse_name = get_timelapse_name(self.cur_shapes_layer.name)
                if timelapse_name not in self.timelapses or self.cur_shapes_layer.name not in self.timelapses[timelapse_name]:
                    raise RuntimeError(f"Internal error: unknown timelapse or frame name {timelapse_name}")
                old_shape_layer_name = self.cur_shapes_layer.name
                for frame_name in self.timelapses[timelapse_name]:
                    self.organoiDL.update_bboxes_scores(frame_name,
                                                self.viewer.layers[frame_name].data, 
                                                self.viewer.layers[frame_name].properties['confidence'],
                                                self.viewer.layers[frame_name].properties['box_id'],
                                                self.stored_confidences[frame_name],
                                                self.stored_diameters[frame_name]
                                            )
                    bboxes, scores, box_ids = self.organoiDL.apply_params(frame_name, self.confidence, self.min_diameter)
                    self._update_detections(bboxes, scores, box_ids, frame_name)
                self.cur_shapes_layer = self.viewer.layers[old_shape_layer_name]
                self._update_num_organoids(len(self.cur_shapes_layer.data))
                self._update_cur_shapes_layer_label_and_checkbox()
            else:
                self.organoiDL.update_bboxes_scores(self.cur_shapes_layer.name,
                                                self.cur_shapes_layer.data, 
                                                self.cur_shapes_layer.properties['confidence'],
                                                self.cur_shapes_layer.properties['box_id'],
                                                self.stored_confidences[self.cur_shapes_layer.name],
                                                self.stored_diameters[self.cur_shapes_layer.name]
                                            )
                bboxes, scores, box_ids = self.organoiDL.apply_params(self.cur_shapes_layer.name, self.confidence, self.min_diameter)
                self._update_detections(bboxes, scores, box_ids, self.cur_shapes_layer.name)

    def _on_diameter_slider_changed(self):
        """ Is called whenever user changes the Minimum Diameter slider """
        # get current value
        if self.diameter_textbox_changed: return
        self.min_diameter = self.min_diameter_slider.value()
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
            properties = {'box_id': [],'confidence': []}
            text_params = {'string': 'ID: {box_id}\nConf.: {confidence:.2f}',
                        'size': 12,
                        'anchor': 'upper_left',
                        'color': settings.TEXT_COLOR}
            edge_color = settings.COLOR_DEFAULT
            img_data = self.viewer.layers[self.image_layer_name].data

            if self.image_layer_name in self.timelapse_image_layers:
                # Add custom labels for timelapse
                timelapse_name = f'{self.image_layer_name}-Labels-Custom-'
                if not timelapse_name in self.timelapses:
                    self.timelapses[timelapse_name] = []
                    self.cur_timelapse_name = timelapse_name
                    self.timelapse_selection.addItem(timelapse_name)
                    self.timelapse_selection.setCurrentText(timelapse_name)

                from qtpy.QtWidgets import QMessageBox
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
                            self.label2im[frame_layer_name] = self.image_layer_name
                            self.organoiDL.next_id[frame_layer_name] = 0
                            new_layer = self.viewer.add_shapes(
                                name=frame_layer_name,
                                scale=self.viewer.layers[self.image_layer_name].scale[:2],
                                face_color='transparent',
                                properties=properties.copy(),
                                text=text_params,
                                edge_color=edge_color,
                                shape_type='rectangle',
                                edge_width=12
                            )                
                            self.stored_confidences[frame_layer_name] = self.confidence
                            self.stored_diameters[frame_layer_name] = self.min_diameter
                            self.timelapses[timelapse_name].append(frame_layer_name)
                    self._on_frame_change()            
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
                    self.label2im[frame_layer_name] = self.image_layer_name
                    self.organoiDL.next_id[frame_layer_name] = 0
                    new_layer = self.viewer.add_shapes(
                        name=frame_layer_name,
                        scale=self.viewer.layers[self.image_layer_name].scale[:2],
                        face_color='transparent',
                        properties=properties.copy(),
                        text=text_params,
                        edge_color=edge_color,
                        shape_type='rectangle',
                        edge_width=12
                    )
                    self.timelapses[timelapse_name].append(frame_layer_name)
                    self.stored_confidences[frame_layer_name] = self.confidence
                    self.stored_diameters[frame_layer_name] = self.min_diameter
            else:
                self.label2im[new_layer_name] = self.image_layer_name
                self.organoiDL.next_id[new_layer_name] = 0
                new_layer = self.viewer.add_shapes( 
                    name=new_layer_name,
                    scale=self.viewer.layers[self.image_layer_name].scale[:2],
                    face_color='transparent',  
                    properties = properties,
                    text = text_params,
                    edge_color=edge_color,
                    shape_type='rectangle',
                    edge_width=12
                )
                self.stored_confidences[new_layer_name] = self.confidence
                self.stored_diameters[new_layer_name] = self.min_diameter

            self.cur_shapes_layer = new_layer
            
            self._update_num_organoids(len(self.cur_shapes_layer.data))
            self._update_cur_shapes_layer_label_and_checkbox()
            self.cur_shapes_layer.current_edge_width = 12
            self._save_cache_results(self.cur_shapes_layer.name)

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
        self.cur_shapes_layer.mode = 'add_rectangle'

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
        # make sure this stuff isn't done if data in the layer has been changed by the sliders - only by the users
        key = 'napari-organoid-counter:_rerun'
        if key in self.cur_shapes_layer.metadata: 
            return
        
        # get new ids, new boxes and update the number of organoids
        new_ids = self.cur_shapes_layer.properties['box_id']
        new_bboxes = self.cur_shapes_layer.data
        new_scores = self.cur_shapes_layer.properties['confidence']
        if len(new_ids) != len(new_scores):
            show_error('Number of IDs and scores do not match!')
            return
    
        self._update_num_organoids(len(new_ids))
        curr_next_id = self.organoiDL.next_id[self.cur_shapes_layer.name]
        
        # check if duplicate ids
        if len(new_ids) > len(set(new_ids)) or np.isnan(new_ids).any():
            used_id = set()
            for idx, id_val in enumerate(new_ids):
                if id_val in used_id or np.isnan(id_val):
                    new_ids[idx] = int(curr_next_id)
                    used_id.add(curr_next_id)
                    curr_next_id += 1
                    new_scores[idx] = 1.0
                else:
                    used_id.add(id_val)


        new_ids = list(map(int, new_ids))
        self.organoiDL.update_bboxes_scores(self.cur_shapes_layer.name, new_bboxes, new_scores, new_ids, 
                                            self.stored_confidences[self.cur_shapes_layer.name], 
                                            self.stored_diameters[self.cur_shapes_layer.name]
                                        )
        self._save_cache_results(self.cur_shapes_layer.name)

        # set new properties to shapes layer
        self.cur_shapes_layer.properties = { 'box_id': new_ids, 'confidence': new_scores }
        # refresh text displayed
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
        hbox.addWidget(run_btn)
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
        export_btn.clicked.connect(self._on_export_click)
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

    def _setup_timelapse_widget(self):
        """
        Sets up the GUI part for timelapse and tracking.
        """
        timelapse_widget = QGroupBox('Timelapse and tracking')
        vbox = QVBoxLayout()
        
        # Timelapse selector
        hbox_selector = QHBoxLayout()
        timelapse_label = QLabel('Timelapse: ', self)
        timelapse_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.timelapse_selection = QComboBox()
        self.timelapse_selection.currentIndexChanged.connect(self._on_timelapse_change)
        hbox_selector.addWidget(timelapse_label, 2)
        hbox_selector.addWidget(self.timelapse_selection, 4)
        vbox.addLayout(hbox_selector)
        
        # Buttons
        hbox_buttons = QHBoxLayout()
        hbox_buttons.addStretch(1)
        create_timelapse_btn = QPushButton("Create labelled timelapse")
        run_tracking_btn = QPushButton("Run Tracking (WIP)")
        delete_timelapse_btn = QPushButton("Delete timelapse")
        delete_timelapse_btn.clicked.connect(self._on_delete_timelapse)
        create_timelapse_btn.clicked.connect(self._on_create_labelled_timelapse)
        run_tracking_btn.clicked.connect(self._on_run_tracking)
        hbox_buttons.addWidget(create_timelapse_btn)
        hbox_buttons.addSpacing(15)
        hbox_buttons.addWidget(run_tracking_btn)
        hbox_buttons.addSpacing(15)
        hbox_buttons.addWidget(delete_timelapse_btn)
        hbox_buttons.addStretch(1)
        vbox.addLayout(hbox_buttons)
        
        timelapse_widget.setLayout(vbox)
        return timelapse_widget
    
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
        from qtpy.QtWidgets import QFileDialog
        import cv2

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

    def _on_run_tracking(self):
        pass

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
        # if layer_names: 
        #     return [] + layer_names
        # else: 
        #     return []

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

    def _on_annotation_labels_layer_change(self):
        """
        Called when user changes layer of labels used for segmentation
        """
        self.label_layer_name = self.annotation_image_layer_selection.currentText()
        # Show or hide the "Run for entire timelapse" checkbox based on layer name
        if self.label_layer_name.startswith("TL_Frame"):
            self.annotation_run_for_timelapse_checkbox.setVisible(True)
        else:
            self.annotation_run_for_timelapse_checkbox.setVisible(False)

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
                top_item.setText(0, f"Detection ID {self.cur_shapes_layer.properties['box_id'][index]}")
                top_item.setExpanded(False)

                # Add properties as child items
                for prop_name, prop_values in self.cur_shapes_layer.properties.items():
                    if prop_name != 'box_id':
                        child_item = QTreeWidgetItem(top_item)
                        child_item.setText(0, prop_name)
                        child_item.setText(1, str(prop_values[index]))
            self.detection_data_tree.expandAll()
        else:
            self.tab_widget.setTabEnabled(1, False)

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
        self.annotation_image_layer_selection = QComboBox()
        if self.image_layer_names is not None:
            for name in self.image_layer_names:
                if not name.startswith('Segmentation-') and not name.startswith('TL_'):
                    self.annotation_image_layer_selection.addItem(name)
        self.annotation_image_layer_selection.currentIndexChanged.connect(self._on_annotation_labels_layer_change)
        hbox_config.addWidget(self.annotation_image_layer_selection, 2)
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

        #Annotation name
        hbox_config0 = QHBoxLayout()
        annotation_name_desc = QLabel('Annotation name: ', self)
        self.new_annotation_name = QLineEdit()
        hbox_config0.addWidget(annotation_name_desc, 1)
        hbox_config0.addWidget(self.new_annotation_name, 4)
        vbox.addLayout(hbox_config0)
        
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
        feature_name_desc = QLabel('Annotation name: ', self)
        self.resume_feature_name = QComboBox()
        for ft in self._load_annotation_features():
            self.resume_feature_name.addItem(ft)
        
        hbox_config1.addWidget(feature_name_desc, 1)
        hbox_config1.addWidget(self.resume_feature_name, 4)
        vbox.addLayout(hbox_config1)

        # Feature action
        hbox_config2 = QHBoxLayout()
        delete_feature = QPushButton('Delete')
        delete_feature.clicked.connect(self._on_delete_annotation_feature)
        delete_feature.setStyleSheet("border: 0px")

        hbox_config2 = QHBoxLayout()
        start_annotation = QPushButton('Continue annotating')
        start_annotation.clicked.connect(self._on_continue_annotation)
        start_annotation.setStyleSheet("border: 0px")

        hbox_config2.addWidget(delete_feature, 2)
        hbox_config2.addWidget(start_annotation, 2)
        # hbox_config2.addSpacing(15)
        # hbox_config2.addStretch(1)   
        vbox.addLayout(hbox_config2) 

        widget.setLayout(vbox)
        return widget

    def _on_create_annotation_feature(self):
        annotation_name = self.new_annotation_name.text()
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
        bboxes = convert_boxes_from_napari_view(bboxes).numpy()
        properties = labels_layer.properties.copy()

        for property_name, property in properties.items():
            if len(property) != bboxes.shape[0]:
                raise RuntimeError(f"Number of properties for propertsave_annotay {property_name} ({len(property)}) doesn't match number of bounding boxes ({bboxes.shape[0]})")
            
        annotation_dialogue = get_annotation_dialogue(image, bboxes, properties, annotation_data, self)
        if annotation_dialogue.exec() != QDialog.Accepted:
            show_warning("Annotation cancelled. But your changes have been saved.")
            return
        new_annotations = annotation_dialogue.get_annotations()
        if annotation_data['property_name'] in properties:
            feature_data = properties[annotation_data['property_name']]
        else:
            feature_data = ["" for i in range(len(properties['box_id']))]
        for box_id, value in new_annotations.items():
            feature_data[int(box_id)] = value
        properties.update({annotation_data['property_name']: feature_data})
        labels_layer.properties = properties

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
