from typing import List

from skimage.io import imsave
from skimage.color import rgb2gray
from datetime import datetime

import napari

from napari import layers
from napari.utils.notifications import show_info, show_error, show_warning
from urllib.request import urlretrieve
from napari_organoid_analyzer._utils import convert_boxes_from_napari_view, collate_instance_masks


import numpy as np

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QApplication, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit, QProgressBar, QSlider, QTabWidget, QTreeWidget, QTreeWidgetItem, QCheckBox

from napari_organoid_analyzer._orgacount import OrganoiDL
from napari_organoid_analyzer import _utils as utils
from napari_organoid_analyzer import settings
import torch
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")


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
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        settings.UTIL_DIR.mkdir(parents=True, exist_ok=True)
        utils.add_local_models()
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
        self.cur_shapes_name = ''
        self.cur_shapes_layer = None
        self.num_organoids = 0
        self.original_images = {}
        self.original_contrast = {}
        self.stored_confidences = {}
        self.stored_diameters = {}
        self.label2im = {}

        # Initialize multi_annotation_mode to False by default
        self.multi_annotation_mode = False
        # self.single_annotation_mode = True  # Initially, it's single annotation mode

        # Setup tab widget
        self.tab_widget = QTabWidget()
        self.configuration_tab = QWidget()
        self.detection_data_tab = QWidget()

        # Add tabs to the tab widget
        self.tab_widget.addTab(self.configuration_tab, "Configuration")
        self.tab_widget.addTab(self.detection_data_tab, "Detection data")
        self.tab_widget.setTabEnabled(1, False)  # Initially disable the "Detection data" tab

        # Set up the layout for the configuration tab
        self.configuration_tab.setLayout(QVBoxLayout())
        self.configuration_tab.layout().addWidget(self._setup_input_widget())
        self.configuration_tab.layout().addWidget(self._setup_output_widget())
        self.configuration_tab.layout().addWidget(self._setup_segmentation_widget())

        # Set up the layout for the detection data tab
        self.detection_data_tab.setLayout(QVBoxLayout())
        self.detection_data_tree = QTreeWidget()
        self.detection_data_tree.setHeaderLabels(["Detections", "Properties"])
        self.detection_data_tab.layout().addWidget(self.detection_data_tree)

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
        self.image_layer_names = self._get_layer_names()
        if len(self.image_layer_names)>0: self._update_added_image(self.image_layer_names)
        self.shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
        if len(self.shape_layer_names)>0: self._update_added_shapes(self.shape_layer_names)
        # and watch for newly added images or shapes
        self.viewer.layers.events.inserted.connect(self._added_layer)
        self.viewer.layers.events.removed.connect(self._removed_layer)
        self.viewer.layers.selection.events.changed.connect(self._sel_layer_changed)
        for layer in self.viewer.layers:
            layer.events.name.connect(self._on_layer_name_change)
    
        # setup flags used for changing slider and text of min diameter and confidence threshold
        self.diameter_slider_changed = False 
        self.confidence_slider_changed = False

        # Key binding to change the edge_color of the bounding boxes to green
        @self.viewer.bind_key('g')
        def change_edge_color_to_green(viewer: napari.Viewer):
            if not self.multi_annotation_mode:  # Check if single-annotation mode is active
                show_error("Cannot change edge color. Change to multi-annotation mode to enable this feature.")
                return
            if self.cur_shapes_layer is not None:  # Ensure shapes layer exists
                selected_shapes = self.cur_shapes_layer.selected_data # Retrieves indices of shapes currently selected, returns a set 
                if len(selected_shapes) > 0:
                    # Modify the edge color only for the selected shapes
                    current_edge_colors = self.cur_shapes_layer.edge_color 
                    for idx in selected_shapes:
                        # Save original color
                        # if idx not in self.original_colors: 
                            # self.original_colors[idx] = current_edge_colors[idx].copy()
                        # Update to the new color
                        current_edge_colors[idx] = settings.COLOR_CLASS_1
                    self.cur_shapes_layer.edge_color = current_edge_colors  # Apply the changes
                    show_info(f"Changed edge color of shapes {list(selected_shapes)} to green.")
                else:
                    show_warning("No shapes selected to change edge color.")

        # Key binding to change the edge_color of the bounding boxes to blue
        @self.viewer.bind_key('h')
        def change_edge_color_to_blue(viewer: napari.Viewer):
            if not self.multi_annotation_mode:  # Check if single-annotation mode is active
                show_error("Cannot change edge color. Change to multi-annotation mode to enable this feature.")
                return         
            if self.cur_shapes_layer is not None:  # Ensure shapes layer exists
                selected_shapes = self.cur_shapes_layer.selected_data
                if len(selected_shapes) > 0:
                    # Modify the edge color only for the selected shapes
                    current_edge_colors = self.cur_shapes_layer.edge_color
                    for idx in selected_shapes:
                        # Save original color
                        # if idx not in self.original_colors: 
                            # self.original_colors[idx] = current_edge_colors[idx].copy()
                        # Update to the new color
                        current_edge_colors[idx] = settings.COLOR_CLASS_2
                    self.cur_shapes_layer.edge_color = current_edge_colors  # Apply the changes
                    show_info(f"Changed edge color of {list(selected_shapes)} to blue.")
                else:
                    show_warning("No shapes selected to change edge color.")

        # Key binding to reset_on_layer_on_layer the edge_color of selected bounding boxes to the original magenta color
        @self.viewer.bind_key('m')
        def change_to_original_color(viewer: napari.Viewer):
            if not self.multi_annotation_mode:  # Check if single-annotation mode is active
                show_info("Cannot change edge color. Change to multi-annotation mode to enable this feature.")
                return
            if self.cur_shapes_layer is not None:  # Ensure shapes layer exists
                selected_shapes = self.cur_shapes_layer.selected_data
                if len(selected_shapes) > 0:
                    current_edge_colors = self.cur_shapes_layer.edge_color
                    # Modify the edge color only for the selected shapes
                    current_edge_colors = self.cur_shapes_layer.edge_color
                    for idx in selected_shapes:
                        # if idx in self.original_colors:
                            # Revert to the original color
                            current_edge_colors[idx] = settings.COLOR_DEFAULT
                    self.cur_shapes_layer.edge_color = current_edge_colors  # Apply the changes
                    show_info(f"Reset edge color of {list(selected_shapes)} to magenta.")
                else:
                    show_warning("No shapes selected to reset edge color.")


    def handle_progress(self, blocknum, blocksize, totalsize):
        """ When the model is being downloaded, this method is called and th progress of the download
        is calculated and displayed on the progress bar. This function was re-implemented from:
        https://www.geeksforgeeks.org/pyqt5-how-to-automate-progress-bar-while-downloading-using-urllib/ """
        read_data = blocknum * blocksize # calculate the progress
        if totalsize > 0:
            download_percentage = read_data * 100 / totalsize
            self.progress_bar.setValue(int(download_percentage))
            QApplication.processEvents()

    def _sel_layer_changed(self, event):
        """ Is called whenever the user selects a different layer to work on. """
        cur_layer_list = list(self.viewer.layers.selection)
        if len(cur_layer_list)==0: return
        cur_seg_selected = cur_layer_list[-1]
        if cur_seg_selected.name == self.cur_shapes_name: return
        # switch to values of other shapes layer if clicked
        if type(cur_seg_selected)==layers.Shapes:
            print(cur_seg_selected, self.cur_shapes_layer, self.stored_confidences, self.stored_diameters)
            if self.cur_shapes_layer is not None:
                self.stored_confidences[self.cur_shapes_name] = self.confidence_slider.value()/100
                self.stored_diameters[self.cur_shapes_name] = self.min_diameter_slider.value()
            self.cur_shapes_layer = cur_seg_selected
            self.cur_shapes_name = cur_seg_selected.name
            # update min diameter text and slider with previous value of that layer
            self.min_diameter = self.stored_diameters[self.cur_shapes_name]
            self.min_diameter_textbox.setText(str(self.min_diameter))
            self.min_diameter_slider.setValue(self.min_diameter)
            # update confidence text and slider with previous value of that layer
            self.confidence = self.stored_confidences[self.cur_shapes_name]
            self.confidence_textbox.setText(str(self.confidence))
            self.confidence_slider.setValue(int(self.confidence*100))

    def _added_layer(self, event):
        # get names of added layers, image and shapes
        new_image_layer_names = self._get_layer_names()
        new_shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
        new_image_layer_names = [name for name in new_image_layer_names if name not in self.image_layer_names]
        new_shape_layer_names = [name for name in new_shape_layer_names if name not in self.shape_layer_names]
        if len(new_image_layer_names)>0 : 
            self._update_added_image(new_image_layer_names)
            self.image_layer_names = self._get_layer_names()
        if len(new_shape_layer_names)>0:
            self._update_added_shapes(new_shape_layer_names)
            self.shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)

        for layer in self.viewer.layers:
            layer.events.name.connect(self._on_layer_name_change)
            if type(layer) == layers.Shapes:
                layer.events.highlight.connect(self._on_shape_selected)

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

    def _preprocess(self):
        """ Preprocess the current image in the viewer to improve visualisation for the user """
        img = self.original_images[self.image_layer_name]
        img = utils.apply_normalization(img)
        self.viewer.layers[self.image_layer_name].data = img
        self.viewer.layers[self.image_layer_name].contrast_limits = (0,255)

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
        # or if this is the first run
        else:
            # if no organoids were found just make an empty shapes layer
            if self.num_organoids==0: 
                self.cur_shapes_layer = self.viewer.add_shapes(name=labels_layer_name,
                                                               properties={'box_id': [],'confidence': []})
            # otherwise make the layer and add the boxes
            else:
                properties = {'box_id': box_ids,'confidence': scores}
                text_params = {'string': 'ID: {box_id}\nConf.: {confidence:.2f}',
                               'size': 12,
                               'anchor': 'upper_left',
                               'color': settings.TEXT_COLOR}
                self.cur_shapes_layer = self.viewer.add_shapes(bboxes, 
                                                               name=labels_layer_name,
                                                               scale=self.viewer.layers[self.image_layer_name].scale,
                                                               face_color='transparent',  
                                                               properties = properties,
                                                               text = text_params,
                                                               edge_color=settings.COLOR_DEFAULT,
                                                               shape_type='rectangle',
                                                               edge_width=12) # warning generated here
                            
            # set current_edge_width so edge width is the same when users annotate - doesnt' fix new preds being added!
            self.viewer.layers[labels_layer_name].current_edge_width = 12
            

    def _on_preprocess_click(self):
        """ Is called whenever preprocess button is clicked """
        if not self.image_layer_name: show_info('Please load an image first and try again!')
        else: self._preprocess()

    def _on_run_click(self):
        """ Is called whenever Run Organoid Counter button is clicked """
        # check if an image has been loaded
        if not self.image_layer_name: 
            show_info('Please load an image first and try again!')
            return
        # check if SAM model exists locally and if not ask user if it's ok to download
        if not utils.return_is_file(settings.UTIL_DIR, settings.SAM_MODEL["filename"]): 
            confirm_window = ConfirmSamUpload(self)
            confirm_window.exec_()
            # if user clicks cancel return doing nothing 
            if confirm_window.result() != QDialog.Accepted: return
            # otherwise download model and display progress in progress bar
            else: 
                self.progress_box.show()
                save_loc = os.path.join(str(settings.UTIL_DIR),  settings.SAM_MODEL["filename"])
                urlretrieve(settings.SAM_MODEL["url"], save_loc, self.handle_progress)
                self.progress_box.hide()

        # check if model exists locally and if not ask user if it's ok to download
        if not utils.return_is_file(settings.MODELS_DIR, settings.MODELS[self.model_name]["filename"]): 
            confirm_window = ConfirmUpload(self, self.model_name)
            confirm_window.exec_()
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
            self.organoiDL.set_scale(self.viewer.layers[self.image_layer_name].scale)
        
        # make sure the number of windows and downsamplings are the same
        if len(self.window_sizes) != len(self.downsampling): 
            show_info('Keep number of window sizes and downsampling the same and try again!')
            return
        
        # get the current image 
        img_data = self.viewer.layers[self.image_layer_name].data
        
        # check if the image is not grayscale and convert it
        if img_data.ndim == 3:
            if img_data.shape[2] == 4:
                img_data = img_data[:, :, :3]
            img_data = rgb2gray(img_data)
            img_data = (img_data * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        
        # update the viewer with the new bboxes
        labels_layer_name = 'Labels-' + self.model_name + '-' + self.image_layer_name + datetime.strftime(datetime.now(), "%H:%M:%S")
        self.label2im[labels_layer_name] = self.image_layer_name
        self.stored_confidences[labels_layer_name] = self.confidence_slider.value()/100
        self.stored_diameters[labels_layer_name] = self.min_diameter_slider.value()
        if labels_layer_name in self.shape_layer_names:
            show_info('Found existing labels layer. Please remove or rename it and try again!')
            return 
        
        # show activity docker for progress bar while running 
        self.viewer.window._status_bar._toggle_activity_dock(True)
       
        # run inference
        self.organoiDL.run(img_data, 
                           labels_layer_name,
                           self.window_sizes,
                           self.downsampling,
                           self.window_overlap)
        
        # set the confidence threshold, remove small organoids and get bboxes in format to visualize
        bboxes, scores, box_ids = self.organoiDL.apply_params(labels_layer_name, self.confidence, self.min_diameter)
        # hide activity dock on completion
        self.viewer.window._status_bar._toggle_activity_dock(False)
        # update widget with results
        self._update_detections(bboxes, scores, box_ids, labels_layer_name)
        # and update cur_shapes_name to newly created shapes layer
        self.cur_shapes_name = labels_layer_name
        # preprocess the image if not done so already to improve visualization
        self._preprocess()

    def _on_run_segmentation(self):
        """
        Is called whether run_segmentation button is clicked
        """
        if not self.label_layer_name:
            show_error("No label layer selected. Please select a label layer and try again.")
            return

        labels_layer = self.viewer.layers[self.label_layer_name]

        if labels_layer is None:
            show_error(f"Layer '{self.label_layer_name}' not found in the viewer.")
            return
        
        bboxes = convert_boxes_from_napari_view(labels_layer.data)

        if not self.label2im[self.label_layer_name] in self.viewer.layers:
            show_error(f"Image layer '{self.label2im[self.label_layer_name]}' not found in the viewer. Please upload the image again")
            return

        image = self.viewer.layers[self.label2im[self.label_layer_name]].data
        if image.shape[2] == 4:
            image = image[:, :, :3]
        segmentation_layer_name = f"Segmentation-{self.label_layer_name}-{datetime.strftime(datetime.now(), '%H:%M:%S')}"
        
        masks, features = self.organoiDL.run_segmentation(image, self.label_layer_name, bboxes)
        
        self.viewer.add_image(collate_instance_masks(masks, color=True), name=segmentation_layer_name, blending='additive')
        if len(labels_layer.properties['box_id']) != masks.shape[0] or len(labels_layer.properties['confidence']) != masks.shape[0]:
            show_error("Mismatch in number of masks and labels. Please rerun the segmentation.")
            return
        tmp_dict = labels_layer.properties.copy()
        tmp_dict.update(features)
        labels_layer.properties = tmp_dict
        self._update_detection_data_tab()
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
        
        # Open the export dialog
        export_dialog = ExportDialog(self, available_features)
        if export_dialog.exec_() != QDialog.Accepted:
            show_warning("Export canceled.")
            return
        
        export_path = export_dialog.get_export_path()
        if not export_path:
            show_error("No export folder selected.")
            return
        
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
            show_info(f"Export completed successfully to {export_path}\nExported: {', '.join(exported_items)}")
        else:
            show_warning("No items were selected for export.")

    def _export_bboxes(self, label_layer, export_path):
        """Export bounding boxes to JSON file"""
        bboxes = label_layer.data
        
        if len(bboxes) == 0: 
            show_warning('No organoids detected! Skipping bounding box export.')
            return
        
        # Check for multi-annotation mode
        if self.multi_annotation_mode:
            # Get the edge colors for all bounding boxes
            edge_colors = label_layer.edge_color
            labels = []

            # Check if all bounding boxes have their edge color set
            green = np.array(settings.COLOR_CLASS_1)
            blue = np.array(settings.COLOR_CLASS_2)

            all_colored = True
            for edge_color in edge_colors:
                if not (np.allclose(edge_color[:3], green[:3]) or np.allclose(edge_color[:3], blue[:3])):
                    all_colored = False
                    break

            if not all_colored:
                show_warning('Not all bounding boxes have a color assigned. Using default labels.')
                labels = [0] * len(bboxes)
            else:
                # Assign organoid label based on edge_color
                for edge_color in edge_colors:
                    if np.allclose(edge_color[:3], green[:3]):
                        labels.append(0)  # Label for green
                    elif np.allclose(edge_color[:3], blue[:3]):
                        labels.append(1)  # Label for blue
                    else:
                        labels.append(0)  # Default label
        else:
            # Single annotation mode: all bounding boxes get a default label
            labels = [0] * len(bboxes)

        data_json = utils.get_bboxes_as_dict(
            bboxes, 
            label_layer.properties['box_id'],
            label_layer.properties['confidence'],
            label_layer.scale,
            labels=labels
        )
            
        # Write bbox coordinates to json
        json_file_path = os.path.join(export_path, f"{self.label_layer_name}_bboxes.json")
        utils.write_to_json(json_file_path, data_json)

    def _export_instance_masks(self, label_layer, export_path):
        """Export instance masks to NPY"""
        
        instance_masks = self.organoiDL.pred_masks[self.label_layer_name]
        if len(instance_masks) == 0:
            show_warning("No masks found for segmentation. Skipping mask export.")
            return
        
        # Export instance masksy
        box_ids = label_layer.properties['box_id']
        mask_dict = {int(box_ids[i]): instance_masks[i] for i in range(len(instance_masks))}
            
        instance_mask_file_path = os.path.join(export_path, f"{self.label_layer_name}_instance_masks.npy")
        np.save(instance_mask_file_path, mask_dict)

    def _export_collated_masks(self, label_layer, export_path):
        """Export collated mask to NPY"""
        
        instance_masks = self.organoiDL.pred_masks[self.label_layer_name]
        if len(instance_masks) == 0:
            show_warning("No masks found for segmentation. Skipping mask export.")
            return

        collated_mask = collate_instance_masks(instance_masks)
        collated_mask_file_path = os.path.join(export_path, f"{self.label_layer_name}_collated_mask.npy")
        np.save(collated_mask_file_path, collated_mask)

    def _export_features(self, label_layer, export_path, selected_features):
        """Export selected features to CSV"""
        # Extract only the selected features
        features_to_export = {}
        for feature, selected in selected_features.items():
            if selected and feature in label_layer.properties:
                features_to_export[feature] = label_layer.properties[feature]
        
        # Convert to pandas DataFrame
        if features_to_export:
            df = pd.DataFrame(features_to_export)
            features_file_path = os.path.join(export_path, f"{self.label_layer_name}_features.csv")
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
        if fd.exec_():
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
        if self.organoiDL.img_scale[0]==0: self.organoiDL.set_scale(self.cur_shapes_layer.scale)
        
        # make sure to add info to cur_shapes_layer.metadata to differentiate this action from when user adds/removes boxes
        with utils.set_dict_key( self.cur_shapes_layer.metadata, 'napari-organoid-counter:_rerun', True):
            # first update bboxes in organoiDLin case user has added/removed
            self.organoiDL.update_bboxes_scores(self.cur_shapes_name,
                                                self.cur_shapes_layer.data, 
                                                self.cur_shapes_layer.properties['confidence'],
                                                self.cur_shapes_layer.properties['box_id'])
            # and get new boxes, scores and box ids based on new confidence and min_diameter values 
            bboxes, scores, box_ids = self.organoiDL.apply_params(self.cur_shapes_name, self.confidence, self.min_diameter)
            self._update_detections(bboxes, scores, box_ids, self.cur_shapes_name)

    def _on_diameter_slider_changed(self):
        """ Is called whenever user changes the Minimum Diameter slider """
        # get current value
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
        if self.min_diameter_slider.value() != self.min_diameter:
            self.min_diameter_slider.setValue(self.min_diameter)
        if len(self.shape_layer_names)==0: return
        self._rerun()

    def _on_confidence_slider_changed(self):
        """ Is called whenever user changes the confidence slider """
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
        if self.confidence_slider.value() != slider_conf_value:
            self.confidence_slider.setValue(slider_conf_value)
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

    def on_annotation_mode_changed(self, index):
        """Callback for dropdown selection."""
        if index == 0:  # Single Annotation
            self.multi_annotation_mode = False
            # self.single_annotation_mode = True
            show_info("Switched to Single Annotation mode.")
        elif index == 1:  # Multi Annotation
            self.multi_annotation_mode = True
            # self.single_annotation_mode = False
            show_info("Switched to Multi Annotation mode.")

    def _on_custom_labels_click(self):
        """
        Called when user clicks on button to add custom organoid annotation to image
        """
        if not self.image_layer_name: 
            show_error('Cannot assign custom label to image. Please load an image first!')
            return
        if self.organoiDL.img_scale[0] == 0:
            self.organoiDL.set_scale(self.viewer.layers[self.image_layer_name].scale)
        new_layer_name = f'Labels-Custom-{self.image_layer_name}-{datetime.strftime(datetime.now(), "%H:%M:%S")}'
        self.label2im[new_layer_name] = self.image_layer_name
        self.stored_confidences[new_layer_name] = self.confidence_slider.value()/100
        self.stored_diameters[new_layer_name] = self.min_diameter_slider.value()
        self.organoiDL.next_id[new_layer_name] = 0

        properties = {'box_id': [],'confidence': []}
        text_params = {'string': 'ID: {box_id}\nConf.: {confidence:.2f}',
                        'size': 12,
                        'anchor': 'upper_left',
                        'color': settings.TEXT_COLOR}
        self.cur_shapes_layer = self.viewer.add_shapes( 
                    name=new_layer_name,
                    scale=self.viewer.layers[self.image_layer_name].scale,
                    face_color='transparent',  
                    properties = properties,
                    text = text_params,
                    edge_color=settings.COLOR_DEFAULT,
                    shape_type='rectangle',
                    edge_width=12)
        self.cur_shapes_layer.current_edge_width = 12

    def _update_added_image(self, added_items):
        """
        Update the selection box with new images if images have been added and update the self.original_images and self.original_contrast dicts.
        Set the latest added image to the current working image (self.image_layer_name)
        """
        for layer_name in added_items:
            self.image_layer_selection.addItem(layer_name)
            self.original_images[layer_name] = self.viewer.layers[layer_name].data
            self.original_contrast[layer_name] = self.viewer.layers[self.image_layer_name].contrast_limits
        self.image_layer_name = added_items[0]
        self.image_layer_selection.setCurrentText(self.image_layer_name)

    def _update_removed_image(self, removed_layers):
        """
        Update the selection box by removing image names if image has been deleted and remove items from self.original_images and self.original_contrast dicts.
        """
        # update drop-down selection box and remove image from dict
        for removed_layer in removed_layers:
            item_id = self.image_layer_selection.findText(removed_layer)
            if item_id >= 0:
                self.image_layer_selection.removeItem(item_id)
            self.original_images.pop(removed_layer)
            self.original_contrast.pop(removed_layer)

    def _update_added_shapes(self, added_items):
        """
        Update the selection box by shape layer names if it they have been added, update current working shape layer and instantiate OrganoiDL if not already there
        """
        # update the drop down box displaying shape layer names for saving
        for idx, layer_name in enumerate(added_items):
            self.segmentation_image_layer_selection.addItem(layer_name)
        # set the latest added shapes layer to the shapes layer that has been selected for saving and visualisation
        self.cur_shapes_name = added_items[0]
        self.cur_shapes_layer = self.viewer.layers[self.cur_shapes_name]
        # get the bounding box and update the displayed number of organoids
        self._update_num_organoids(len(self.cur_shapes_layer.data)) 
        # listen for a data change in the current shapes layer
        self.organoiDL.update_bboxes_scores(self.cur_shapes_name,
                                            self.cur_shapes_layer.data,
                                            self.cur_shapes_layer.properties['confidence'],
                                            self.cur_shapes_layer.properties['box_id']
                                            )
        self.cur_shapes_layer.events.data.connect(self.shapes_event_handler)
        
    def _update_remove_shapes(self, removed_layers):
        """
        Update the selection box by removing shape layer names if it they been deleted and set 
        """
        # update selection box by removing image names if image has been deleted       
        for removed_layer in removed_layers:
            item_id = self.segmentation_image_layer_selection.findText(removed_layer)
            self.segmentation_image_layer_selection.removeItem(item_id)
            self.label2im.pop(removed_layer, None)
            self.stored_confidences.pop(removed_layer, None)
            self.stored_diameters.pop(removed_layer, None)
            if removed_layer==self.cur_shapes_name: 
                self._update_num_organoids(0)
            self.organoiDL.remove_shape_from_dict(removed_layer)

    def shapes_event_handler(self, event):
        """
        This function will be called every time the current shapes layer data changes
        """
        # make sure this stuff isn't done if data in the layer has been changed by the sliders - only by the users
        key = 'napari-organoid-counter:_rerun'
        if key in self.cur_shapes_layer.metadata: 
            return
        
        # get new ids, new boxes and update the number of organoids
        new_ids = self.viewer.layers[self.cur_shapes_name].properties['box_id']
        new_bboxes = self.viewer.layers[self.cur_shapes_name].data
        new_scores = self.viewer.layers[self.cur_shapes_name].properties['confidence']
        if len(new_ids) != len(new_scores):
            print('[ERROR] Number of IDs and scores do not match!')
            show_error('Number of IDs and scores do not match!')
            return
    
        self._update_num_organoids(len(new_ids))
        curr_next_id = self.organoiDL.next_id[self.cur_shapes_name]
        
        # check if duplicate ids
        if len(new_ids) > len(set(new_ids)) or np.isnan(new_ids).any():
            used_id = set()
            for idx, id_val in enumerate(new_ids):
                if id_val in used_id or np.isnan(id_val):
                    new_ids[idx] = curr_next_id
                    used_id.add(curr_next_id)
                    curr_next_id += 1
                    new_scores[idx] = 1.0
                else:
                    used_id.add(id_val)
        
        print(new_ids, new_bboxes, new_scores)
        self.organoiDL.update_bboxes_scores(self.cur_shapes_name, new_bboxes, new_scores, new_ids)

        # set new properties to shapes layer
        self.viewer.layers[self.cur_shapes_name].properties = { 'box_id': new_ids, 'confidence': new_scores }
        # refresh text displayed
        self.viewer.layers[self.cur_shapes_name].refresh()
        self.viewer.layers[self.cur_shapes_name].refresh_text()

    def _setup_input_widget(self):
        """
        Sets up the GUI part which corresposnds to the input configurations
        """
        # setup all the individual boxes
        input_box = self._setup_input_box()
        model_box = self._setup_model_box()
        window_sizes_box = self._setup_window_sizes_box()
        downsampling_box = self._setup_downsampling_box()
        run_box = self._setup_run_box()
        annotation_mode_box = self._setup_annotation_mode_box() # Annotation mode dropdown to select single or multi-annotation
        self._setup_progress_box()

        # and add all these to the layout
        input_widget = QGroupBox('Input configurations')
        vbox = QVBoxLayout()
        vbox.addLayout(input_box)
        vbox.addLayout(model_box)
        vbox.addLayout(window_sizes_box)
        vbox.addLayout(downsampling_box)
        vbox.addLayout(run_box)
        vbox.addLayout(annotation_mode_box)  # Add the annotation dropdown
        vbox.addWidget(self.progress_box)
        input_widget.setLayout(vbox)
        return input_widget

    def _setup_output_widget(self):
        """
        Sets up the GUI part which corresposnds to the parameters and outputs
        """
        # setup all the individual boxes
        self.organoid_number_label = QLabel('Number of organoids: '+str(self.num_organoids), self)
        self.organoid_number_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # and add all these to the layout
        output_widget = QGroupBox('Parameters and outputs')
        vbox = QVBoxLayout()
        vbox.addLayout(self._setup_min_diameter_box())
        vbox.addLayout(self._setup_confidence_box() )
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
                if not name.startswith('Segmentation-'):
                    self.image_layer_selection.addItem(name)
        #self.image_layer_selection.setItemText(self.image_layer_name)
        self.image_layer_selection.currentIndexChanged.connect(self._on_image_selection_changed)
        # setup preprocess button to improve visualisation
        preprocess_btn = QPushButton("Preprocess")
        preprocess_btn.clicked.connect(self._on_preprocess_click)
        # and add all these to the layout
        hbox.addWidget(image_label, 2)
        hbox.addWidget(self.image_layer_selection, 4)
        hbox.addWidget(preprocess_btn, 4)
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
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        run_btn = QPushButton("Run Organoid Counter")
        run_btn.clicked.connect(self._on_run_click)
        run_btn.setStyleSheet("border: 0px")
        custom_btn = QPushButton("Add custom labels")
        custom_btn.clicked.connect(self._on_custom_labels_click)
        custom_btn.setStyleSheet("border: 0px")
        hbox.addWidget(run_btn)
        hbox.addStretch(1)
        hbox.addWidget(custom_btn)
        hbox.addStretch(1)
        return hbox
    
    def _setup_annotation_mode_box(self):
        """
        Sets up the GUI part where the annotation mode is selected.
        """
        hbox = QHBoxLayout()

        # Label
        annotation_mode_label = QLabel("Annotation Mode:", self)
        hbox.addWidget(annotation_mode_label)

        # Dropdown
        self.annotation_mode_dropdown = QComboBox()
        self.annotation_mode_dropdown.addItems(["Single Annotation", "Multi Annotation"])
        self.annotation_mode_dropdown.currentIndexChanged.connect(self.on_annotation_mode_changed)
        hbox.addWidget(self.annotation_mode_dropdown)
        
        return hbox

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
                if not name.startswith('Segmentation-'):
                    self.segmentation_image_layer_selection.addItem(name)
        self.segmentation_image_layer_selection.currentIndexChanged.connect(self._on_labels_layer_change)
        hbox_img.addWidget(image_label, 2)
        hbox_img.addWidget(self.segmentation_image_layer_selection, 4)
        vbox.addLayout(hbox_img)
        
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
        segmentation_widget.setLayout(vbox)
        return segmentation_widget

    def _get_layer_names(self, layer_type: layers.Layer = layers.Image) -> List[str]:
        """
        Get a list of layer names of a given layer type.
        """
        layer_names = [layer.name for layer in self.viewer.layers if type(layer) == layer_type]
        if layer_names: return [] + layer_names
        else: return []

    def _on_labels_layer_change(self):
        """
        Called when user changes layer of labels used for segmentation
        """
        self.label_layer_name = self.segmentation_image_layer_selection.currentText()
    
    def _on_layer_name_change(self, event):
        """
        Called whether user changes the name of any of the layers.
        """
        
        # Update selectors for image and shapes layers
        self.segmentation_image_layer_selection.clear()
        for name in self._get_layer_names(layer_type=layers.Shapes): 
            self.segmentation_image_layer_selection.addItem(name)

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

class ExportDialog(QDialog):
    """
    Dialog for selecting export options
    """
    def __init__(self, parent, available_features):
        super().__init__(parent)
        self.setWindowTitle("Export Options")
        self.setMinimumWidth(500)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Export path selection
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Export to folder:"))
        self.path_input = QLineEdit()
        path_layout.addWidget(self.path_input, 1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_folder)
        path_layout.addWidget(browse_button)
        layout.addLayout(path_layout)
        
        # What to export
        layout.addWidget(QLabel("Select what to export:"))
        
        # Export options layout (left side of the bottom part)
        options_layout = QVBoxLayout()
        
        # Checkboxes for export options
        self.export_bboxes = QCheckBox("Bounding boxes (JSON)")
        self.export_bboxes.setChecked(True)
        self.export_instance_masks = QCheckBox("Instance masks (NPY)")
        self.export_instance_masks.setChecked(True)
        self.export_collated_mask = QCheckBox("Collated mask (NPY)")
        self.export_collated_mask.setChecked(True)
        self.export_features = QCheckBox("Features (CSV)")
        self.export_features.setChecked(True)
        self.export_features.stateChanged.connect(self._toggle_feature_selection)
        
        options_layout.addWidget(self.export_bboxes)
        options_layout.addWidget(self.export_instance_masks)
        options_layout.addWidget(self.export_collated_mask)
        options_layout.addWidget(self.export_features)
        
        # Bottom part with options on left and feature selection on right
        bottom_layout = QHBoxLayout()
        bottom_layout.addLayout(options_layout)
        
        # Feature selection (right side)
        self.feature_selection_widget = QWidget()
        feature_layout = QVBoxLayout()
        feature_layout.addWidget(QLabel("Select features to export:"))
        
        self.feature_checkboxes = {}
        for feature in available_features:
            checkbox = QCheckBox(feature)
            checkbox.setChecked(True)  # Default checked
            self.feature_checkboxes[feature] = checkbox
            feature_layout.addWidget(checkbox)
        
        feature_layout.addStretch()
        self.feature_selection_widget.setLayout(feature_layout)
        bottom_layout.addWidget(self.feature_selection_widget)
        layout.addLayout(bottom_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        export_button = QPushButton("Export")
        export_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(export_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if folder:
            self.path_input.setText(folder)
    
    def _toggle_feature_selection(self, state):
        self.feature_selection_widget.setVisible(state == Qt.Checked)
    
    def get_export_path(self):
        return self.path_input.text()
    
    def get_export_options(self):
        return {
            'bboxes': self.export_bboxes.isChecked(),
            'instance_masks': self.export_instance_masks.isChecked(),
            'collated_mask': self.export_collated_mask.isChecked(),
            'features': self.export_features.isChecked()
        }
    
    def get_selected_features(self):
        return {feature: checkbox.isChecked() 
                for feature, checkbox in self.feature_checkboxes.items()}

class ConfirmUpload(QDialog):
    '''
    The QDialog box that appears when the user selects to run organoid counter
    without having the selected model locally
    Parameters
    ----------
        parent: QWidget
            The parent widget, in this case an instance of OrganoidCounterWidget

    '''
    def __init__(self, parent: QWidget, model_name: str):
        super().__init__(parent)

        self.setWindowTitle("Confirm Download")
        # setup buttons and text to be displayed
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        text = (f"Model {model_name} not found locally. Downloading default model to \n"
                +str(settings.MODELS_DIR)+"\n"
                "This will only happen once. Click ok to continue or \n"
                "cancel if you do not agree. You won't be able to run\n"
                "the organoid counter if you click cancel.")
        # add all to layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel(text))
        hbox = QHBoxLayout()
        hbox.addWidget(ok_btn)
        hbox.addWidget(cancel_btn)
        layout.addLayout(hbox)
        self.setLayout(layout)
        # connect ok and cancel buttons with accept and reject signals
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

class ConfirmSamUpload(ConfirmUpload):
    '''
    The QDialog box that appears when the user selects to run organoid counter
    without having the SAM detection and segmentation model locally
    Parameters
    ----------
        parent: QWidget
            The parent widget, in this case an instance of OrganoidCounterWidget

    '''
    def __init__(self, parent: QWidget):
        super().__init__(parent, model_name="")
        text = ("SAM model not found locally. Downloading default model to \n"
                +str(settings.UTIL_DIR)+"\n"
                "This will only happen once. Click ok to continue or \n"
                "cancel if you do not agree. You won't be able to run\n"
                "the organoid segmentation and detection with SAMOS\n" 
                "if you click cancel. WARNING: The model size is 1.2 GB!")
        self.layout().itemAt(0).widget().setText(text)

