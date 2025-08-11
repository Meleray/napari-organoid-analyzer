import os
from urllib.request import urlretrieve
import numpy as np
from napari.utils import progress

from napari_organoid_analyzer import _utils
from napari_organoid_analyzer import settings

#update_version_in_mmdet_init_file('mmdet', '2.2.0', '2.3.0')
import torch
import mmdet
from mmdet.apis import DetInferencer
from segment_anything import SamPredictor, build_sam_vit_l
from napari_organoid_analyzer._SAMOS.models.detr_own_impl_model import DetectionTransformer
from napari_organoid_analyzer._utils import set_posix_windows, polygon2mask, mask2polygon
import matplotlib.pyplot as plt
import cv2
import sys
import logging
import trackpy as tp
import pandas as pd
import json
import copy
import scipy
from scipy.spatial.distance import cdist
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_SAMOS'))

DENSITY_K_NEIGHBORS = 5  # You can adjust this value as needed

class OrganoiDL():
    '''
    The back-end of the organoid analyzer widget
    Attributes
    ----------
        device: torch.device
            The current device, either 'cpu' or 'gpu:0'
        cur_confidence: float
            The confidence threshold of the model
        cur_min_diam: float
            The minimum diameter of the organoids
        model: frcnn
            The Faster R-CNN model
        img_scale: list of floats
            A list holding the image resolution in x and y
        pred_bboxes: dict
            Each key will be a set of predictions of the model, either past or current, and values will be the numpy arrays 
            holding the predicted bounding boxes
        pred_scores: dict
            Each key will be a set of predictions of the model and the values will hold the confidence of the model for each
            predicted bounding box
        pred_ids: dict
            Each key will be a set of predictions of the model and the values will hold the box id for each
            predicted bounding box
        next_id: dict
            Each key will be a set of predictions of the model and the values will hold the next id to be attributed to a 
            newly added box
    '''
    def __init__(self, handle_progress):
        super().__init__()
        
        self.handle_progress = handle_progress
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = None
        self.model = None
        self.img_scale = [0., 0.]

        # Storage for shape layer data. { "shape_layer_name": 
        #{
        #   "detection_data": {
        #       id: {
        #           "bbox": [x1, y1, x2, y2], 
        #           "score": float,
        #           ---- Other features ----
        #      }
        #   },
        #   "segmentation_data": {
        #      "id": {
        #           "mask": [polygon coordinates], # polygon coordinates of the mask
        #           "{signal_name}_mask": [polygon coordinates], # polygon coordinates of the signal mask
        #       }
        #   }
        #   "image_size": [H, W],
        #   "displayed_ids": [list of IDs that are currently displayed in the layer as str],
        #   "next_id": int (next ID to be attributed to a new box)
        #}
        self.storage = {}
        # self.pred_bboxes = {}
        # self.pred_scores = {}
        # self.pred_masks = {}
        # self.signal_masks = {}
        # self.pred_ids = {}
        # self.next_id = {}
        self.sam_predictor = None

    def set_scale(self, img_scale):
        ''' Set the image scale: used to calculate real box sizes. '''
        self.img_scale = img_scale
    
    def init_sam_predictor(self):
        if self.sam_predictor is None:
            sam_model = build_sam_vit_l(checkpoint=_utils.join_paths(str(settings.MODELS_DIR), settings.SAM_MODEL["filename"]))
            self.sam_predictor = SamPredictor(sam_model=sam_model.to(self.device))

    def set_model(self, model_name):
        ''' Initialise  model instance and load model checkpoint and send to device. '''
        self.init_sam_predictor()
        self.model_name = model_name
        model_checkpoint = _utils.join_paths(str(settings.MODELS_DIR), settings.MODELS[model_name]["filename"])
        if model_name == 'SAMOS':
            with set_posix_windows():
                checkpoint = torch.load(model_checkpoint, map_location=self.device, weights_only=False)
            self.model = DetectionTransformer(**checkpoint['hyper_parameters'])
            new_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                new_key = key.replace('model.', '') 
                new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict)
            self.model.to(self.device)
        else:
            mmdet_path = os.path.dirname(mmdet.__file__)
            config_dst = _utils.join_paths(mmdet_path, str(settings.CONFIGS[model_name]["destination"]))
            # download the corresponding config if it doesn't exist already
            if not os.path.exists(config_dst):
                urlretrieve(settings.CONFIGS[model_name]["source"], config_dst, self.handle_progress)
            self.model = DetInferencer(config_dst, model_checkpoint, self.device, show_progress=False)

    def download_model(self, model_name='yolov3'):
        ''' Downloads the model from zenodo and stores it in settings.MODELS_DIR '''
        # specify the url of the model which is to be downloaded
        down_url = settings.MODELS[model_name]["source"]
        # specify save location where the file is to be saved
        save_loc = _utils.join_paths(str(settings.MODELS_DIR), settings.MODELS[model_name]["filename"])
        # downloading using urllib
        urlretrieve(down_url, save_loc, self.handle_progress)

    def sliding_window(self,
                       test_img,
                       step,
                       window_size,
                       rescale_factor,
                       prepadded_height,
                       prepadded_width,
                       crop_offset,
                       bboxes_list=[],
                       scores_list=[]):
        ''' Runs sliding window inference and returns predicting bounding boxes and confidence scores for each box.
        Inputs
        ----------
        test_img: Tensor of size [B, C, H, W]
            The image ready to be given to model as input
        step: int
            The step of the sliding window, same in x and y
        window_size: int
            The sliding window size, same in x and y
        rescale_factor: float
            The rescaling factor by which the image has already been resized. Is 1/downsampling
        prepadded_height: int
            The image height before padding was applied
        prepadded_width: int
            The image width before padding was applied
        crop_offset: list of int
            The [x_min, y_min] offset of the current crop in the original image.
        bboxes_list: list of
            The
        scores_list: list of
            The
        Outputs
        ----------
        pred_bboxes: list of Tensors, default is an empty list
            The  resulting predicted boxes are appended here - if model is run at different window
            sizes and downsampling this list will store results of all runs of the sliding window
            so will not be empty the second, third etc. time.
        scores_list: list of Tensor, default is an empty list
            The  resulting confidence scores of the model for the predicted boxes are appended here 
            Same as pred_bboxes, can be empty on first run but stores results of all runs.
        '''
        for i in progress(range(0, prepadded_height, step), desc="height"):
            for j in progress(range(0, prepadded_width, step), desc="width"):
                # cro
                img_crop = test_img[i:(i+window_size), j:(j+window_size)]
                # get predictions
                if self.model_name == 'SAMOS':
                    #self.model.eval()
                    with torch.inference_mode():
                        self.sam_predictor.set_image(img_crop)
                        image_embedding = self.sam_predictor.features.to(self.device)
                        pred = self.model.forward(image_embedding, window_size)
                        bboxes = pred[0]['boxes']
                        scores = pred[0]['scores']
                else:
                    output = self.model(img_crop)
                    bboxes = output['predictions'][0]['bboxes']
                    scores = output['predictions'][0]['scores']
                if len(bboxes)==0:
                    print(f"Step ({i},{j}): No predictions")
                    continue
                else:
                    print(f"Step ({i},{j}): {bboxes[0]}, {scores[0]}")
                    for bbox_id in range(len(bboxes)):
                        y1, x1, y2, x2 = bboxes[bbox_id] # predictions from model will be in form x1,y1,x2,y2
                        x1_real = torch.div(x1+i, rescale_factor, rounding_mode='floor') + crop_offset[0]
                        x2_real = torch.div(x2+i, rescale_factor, rounding_mode='floor') + crop_offset[0]
                        y1_real = torch.div(y1+j, rescale_factor, rounding_mode='floor') + crop_offset[1]
                        y2_real = torch.div(y2+j, rescale_factor, rounding_mode='floor') + crop_offset[1]
                        bboxes_list.append(torch.Tensor([x1_real, y1_real, x2_real, y2_real]))
                        scores_list.append(scores[bbox_id])
        print('Number of predictions:', len(bboxes_list))
        print('Number of scores:', len(scores_list))
        return bboxes_list, scores_list

    def run(self, 
            img, 
            shapes_name,
            window_sizes,
            downsampling_sizes,   
            window_overlap, 
            crops):
        ''' Runs inference for an image at multiple window sizes and downsampling rates using sliding window ineference.
        The results are filtered using the NMS algorithm and are then stored to dicts.
        Inputs
        ----------
        img: Numpy array of size [H, W]
            The image ready to be given to model as input
        shapes_name: str
            The name of the new predictions
        window_size: list of ints
            The sliding window size, same in x and y, if multiple sliding window will run mulitple times
        downsampling_sizes: list of ints
            The downsampling factor of the image, list size must match window_size
        window_overlap: float
            The window overlap for the sliding window inference.
        crops: Numpy array of size [B, 4]
            Bounding boxes for areas of interest in the image. If not None, the sliding window will run only on these areas.
        ''' 
        bboxes = []
        scores = []
        # run for all window sizes
        for window_size, downsampling in progress(zip(window_sizes, downsampling_sizes), desc="window conf"):
            # compute the step for the sliding window, based on window overlap
            rescale_factor = 1 / downsampling
            # window size after rescaling
            current_window_size = round(window_size * rescale_factor) # Use a different variable name
            step = round(current_window_size * window_overlap)

            for crop_coords in crops:
                x1, y1, x2, y2 = list(map(int, crop_coords))

                cropped_img = img[x1:x2, y1:y2]

                # prepare image for model - norm, tensor, etc.
                ready_img, prepadded_height, prepadded_width = _utils.prepare_img(cropped_img,
                                                                            step,
                                                                            current_window_size,
                                                                            rescale_factor)
                crop_offset_for_sliding_window = [x1, y1]


                bboxes, scores = self.sliding_window(ready_img,
                                                     step,
                                                     current_window_size, # use the rescaled window size
                                                     rescale_factor,
                                                     prepadded_height,
                                                     prepadded_width,
                                                     crop_offset_for_sliding_window,
                                                     bboxes,
                                                     scores)
        # stack results
        bboxes = torch.stack(bboxes)
        scores = torch.Tensor(scores)
        # apply NMS to remove overlaping boxes
        bboxes, pred_scores = _utils.apply_nms(bboxes, scores)

        detection_data = {int(i+1): {
            'bbox': json.dumps(bboxes[i].tolist()),
            'score': pred_scores[i].item(),
        } for i in range(bboxes.size(0))}

        centers = []
        bbox_ids = []
        for bbox_id, det in detection_data.items():
            bbox = json.loads(det['bbox'])
            y1, x1, y2, x2 = bbox
            center = ((y1 + y2) / 2, (x1 + x2) / 2)
            centers.append(center)
            bbox_ids.append(bbox_id)

        centers = np.array(centers)
        dist_matrix = cdist(centers, centers)

        for i, bbox_id in enumerate(bbox_ids):
            # Exclude self (distance zero)
            dists = np.delete(dist_matrix[i], i)
            if len(dists) >= DENSITY_K_NEIGHBORS:
                closest = np.partition(dists, DENSITY_K_NEIGHBORS-1)[:DENSITY_K_NEIGHBORS]
                avg_dist = float(np.mean(closest))
            elif len(dists) > 0:
                avg_dist = float(np.mean(dists))
            else:
                avg_dist = 0.0
            detection_data[bbox_id]['local_density'] = avg_dist

        self.storage[shapes_name] = {
            'detection_data': detection_data,
            'image_size': list(img.shape[:2]),
            'next_id': bboxes.size(0) + 1,
            'segmentation_data': {},
        }

    def _fill_default_data(self, shapes_name):
        """ Checks that all detections have same set of features"""
        if shapes_name not in self.storage:
            return
        all_keys = {key for det_id, props in self.storage[shapes_name]['detection_data'].items() for key in props.keys()}
        for curr_id in self.storage[shapes_name]['detection_data'].keys():
            for key in all_keys:
                if key not in self.storage[shapes_name]['detection_data'][curr_id]:
                    self.storage[shapes_name]['detection_data'][curr_id][key] = None

    def run_segmentation(self, img, shapes_name, bboxes, signal_fields):
        """
        Runs segmentation pipeline for selected image, based on previously detected bboxes
        Inputs
        ----------
        img: Numpy array of size [H, W, 3]
            The input image
        shapes_name: str
            Name of shape layer
        bbooxes: Numpy array of size [N, 4]
            Array of all predicted bboxes in xyxy format
        signal_field: dict({signal_name: signal_field})
            Optional signal fields for the image
        """
        showed_ids = self.storage.get(shapes_name, {}).get('displayed_ids', [])
        assert len(bboxes) == len(showed_ids), "Number of bboxes must match number of stored displayed IDs"
        signal_masks = {}
        if len(bboxes) == 0:
            pred_mask = np.array([])
        else:
            img = _utils.normalize(img)
            bboxes = torch.Tensor(bboxes)
            bboxes = torch.stack((
                bboxes[:, 1],
                bboxes[:, 0],
                bboxes[:, 3],
                bboxes[:, 2]
            ), dim=1).to(self.device)
            self.sam_predictor.set_image(img)
            bboxes = self.sam_predictor.transform.apply_boxes_torch(bboxes, img.shape[:2])
            pred_mask, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=bboxes,
                multimask_output=False
            )
            pred_mask = np.squeeze(pred_mask.cpu().numpy().astype(np.uint8))
            if len(pred_mask.shape) == 2:
                pred_mask = np.expand_dims(pred_mask, axis=0)

            for signal_name, signal_field in progress(signal_fields.items(), desc="signals"):
                self.sam_predictor.set_image(signal_field)
                signal_mask, _, _ = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=bboxes,
                    multimask_output=False
                )
                signal_mask = np.squeeze(signal_mask.cpu().numpy().astype(np.uint8))
                if len(signal_mask.shape) == 2:
                    signal_mask = np.expand_dims(pred_mask, axis=0)
                assert len(pred_mask) == len(signal_mask)
                signal_masks[signal_name] = signal_mask
        
        features = []
        for idx in range(len(showed_ids)):
            curr_id = showed_ids[idx]
            self.storage[shapes_name]['segmentation_data'][curr_id] = {}
            self.storage[shapes_name]['segmentation_data'][curr_id]['mask'] = json.dumps(mask2polygon(pred_mask[idx]))
            if len(signal_masks) > 0:
                for signal_name in signal_masks.keys():
                    self.storage[shapes_name]['segmentation_data'][curr_id][f'{signal_name}_mask'] = json.dumps(mask2polygon(signal_masks[signal_name][idx]))
            curr_features = self._compute_features(
                pred_mask[idx], 
                curr_id, 
                {signal_name: (signal_mask[idx], signal_fields[signal_name][:, :, 0]) for signal_name, signal_mask in signal_masks.items()}, 
            )
            self.storage[shapes_name]['detection_data'][curr_id].update(curr_features)
            features.append(curr_features)
        self._fill_default_data(shapes_name)
        return pred_mask, signal_masks
    
    def _compute_features(self, mask, org_id, signal_data):
        """
        Computes mask-based features for detected organoids
        
        Inputs
        ----------
        mask: Numpy array of size [H, W]
            The mask of a single organoid detection
        signal_data: Dict({signal_name: (mask, field)})
            Masks and fields for signals
        """

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = [cv2.contourArea(contour) for contour in contours]
        id_selected = np.argmax(area)
        if len(contours) > 1:
            logging.warning(f"Multiple contours found for organoid ID#{org_id}. Using the contour with largest area: {area[id_selected]}")
        perimeter = cv2.arcLength(contours[id_selected], True)
        if perimeter > 0:
            roundness = (4 * np.pi * area[id_selected]) / (perimeter ** 2)
        else:
            roundness = 0
        features = {
            'Area (pixel units)': float(area[id_selected]),
            'Perimeter (pixel units)': float(perimeter),
            'Roundness': float(roundness)
        }
        for signal_name, (signal_mask, signal_field) in signal_data.items():
            signal_contours, _ = cv2.findContours(signal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_signal_area = sum([cv2.contourArea(contour) for contour in signal_contours])
            signal_total_intensity = np.sum(signal_mask * signal_field) / 255.0
            signal_mean_intensity = signal_total_intensity / total_signal_area
            features.update({
                f"{signal_name} area (pixel units)": float(total_signal_area),
                f"{signal_name} mean intensity": float(signal_mean_intensity),
                f"{signal_name} total intensity": float(signal_total_intensity),
                #f"{signal_name} components": int(len(signal_contours))
            })
        return features


    def apply_params(self, shapes_name, confidence, min_diameter_um):
        """ After results have been stored in dict this function will filter the dicts based on the confidence
        and min_diameter_um thresholds for the given results defined by shape_name and return the filtered dicts. """
        properties = {key: [] for value in self.storage[shapes_name]['detection_data'].values() for key in value.keys()}
        properties['bbox_id'] = []
        selected_bboxes = []
        min_diameter_x = min_diameter_um / self.img_scale[0]
        min_diameter_y = min_diameter_um / self.img_scale[1]
        for bbox_id in self.storage[shapes_name]['detection_data'].keys():
            bbox = json.loads(self.storage[shapes_name]['detection_data'][bbox_id]['bbox'])
            score = self.storage[shapes_name]['detection_data'][bbox_id]['score']
            if score < confidence: 
                continue
            dx, dy = _utils.get_diams(torch.Tensor(bbox))
            if (dx < min_diameter_x or dy < min_diameter_y) and score < 1:
                continue
            for key in properties.keys():
                if key == 'bbox_id':
                    properties[key].append(int(bbox_id))
                else:
                    properties[key].append(self.storage[shapes_name]['detection_data'][bbox_id].get(key, None))
            selected_bboxes.append(bbox)
        self.storage[shapes_name]['displayed_ids'] = properties['bbox_id'].copy()
        selected_bboxes = _utils.convert_boxes_to_napari_view(selected_bboxes)
        return selected_bboxes, properties

    def update_bboxes_scores(self, shapes_name, new_bboxes, new_properties, image_shape):
        ''' Updated the results dicts, self.pred_bboxes, self.pred_scores and self.pred_ids with new results.
        If the shapes name doesn't exist as a key in the dicts the results are added with the new key. If the
        key exists then new_bboxes, new_scores and new_ids are compared to the class result dicts and the dicts 
        are updated, either by adding some box (user added box) or removing some box (user deleted a prediction).'''
        new_bboxes = _utils.convert_boxes_from_napari_view(new_bboxes)
        new_ids = list(map(int, list(new_properties.pop('bbox_id'))))
        # if run hasn't been run
        if shapes_name not in self.storage.keys():

            self.storage[shapes_name] = {
                "detection_data": {},
                "image_size": image_shape[:2],
                "displayed_ids": new_ids.copy(),
                "next_id": max(new_ids) + 1 if len(new_ids) > 0 else 1
            }
            for idx, box_id in enumerate(new_ids):
                self.storage[shapes_name]['detection_data'][box_id] = {key: new_properties[key][idx] for key in new_properties.keys()}
                self.storage[shapes_name]['detection_data'][box_id]['bbox'] = json.dumps(new_bboxes[idx])

        elif len(new_ids)==0: return

        else:
            old_ids = self.storage[shapes_name]['displayed_ids']
            removed_ids = [old_id for old_id in old_ids if old_id not in new_ids]
            for idx, box_id in enumerate(new_ids):
                self.storage[shapes_name]['detection_data'][box_id] = {key: new_properties[key][idx] for key in new_properties.keys()}
                self.storage[shapes_name]['detection_data'][box_id]['bbox'] = json.dumps(new_bboxes[idx])
            for box_id in removed_ids:
                self.storage[shapes_name]['detection_data'].pop(box_id, None)
            self.storage[shapes_name]['displayed_ids'] = new_ids.copy()
            self.storage[shapes_name]['next_id'] = max(list(map(int, self.storage[shapes_name]['detection_data'].keys())), default=0) + 1
        self._fill_default_data(shapes_name)


    def remove_shape_from_dict(self, shapes_name):
        """ Removes results of shapes_name from all result dicts. """
        self.storage.pop(shapes_name, None)

    def replace_detection_id(self, shape_name, old_id, new_id):
        """
        Replaces the detection ID in the storage dict for the given shape_name.
        Inputs
        ----------
        shape_name: str
            The name of the shape layer to update
        old_id: int
            The old ID to replace
        new_id: int
            The new ID to set
        """
        new_id = int(new_id)
        if shape_name not in self.storage:
            raise ValueError(f"Shape {shape_name} not found in storage.")
        if old_id not in self.storage[shape_name]['detection_data']:
            raise ValueError(f"Old ID {old_id} not found in shape {shape_name} detection data.")
        self.storage[shape_name]['detection_data'][new_id] = self.storage[shape_name]['detection_data'].pop(old_id)
        if old_id in self.storage[shape_name]['segmentation_data']:
            self.storage[shape_name]['segmentation_data'][new_id] = self.storage[shape_name]['segmentation_data'].pop(old_id)
        displayed_ids = self.storage[shape_name]['displayed_ids']
        if old_id in displayed_ids:
            displayed_ids[displayed_ids.index(old_id)] = new_id
        self.storage[shape_name]['next_id'] = max(new_id + 1, self.storage[shape_name]['next_id'])

    def run_tracking(self, imgs, shape_names, tracking_method='trackpy', tracking_params=None):
        """
        Runs tracking for all shapes in shape_names using the tracking model and parameters.
        Updates self.pred_ids for each frame so that IDs are consistent across frames.
        Inputs
        ----------
        imgs: list of Numpy arrays
            List of images to run tracking on (not used for tracking, but kept for API compatibility)
        shape_names: list of str
            List of shape names to run tracking on (should be ordered by frame)
        tracking_method: str
            Identifier of the tracking library to use. Only 'trackpy' is supported.
        tracking_params: dict
            Parameters for the tracking model. Should include:
            - 'search_range': int, maximum distance particles can move between frames
            - 'memory': int, number of frames a particle can be missed before being discarded
            - 'create_missing_detections': bool, whether to create new bounding boxes for 
              tracked objects that are missing in current frame based on memory parameter (default: False)
        """
        if tracking_method != 'trackpy':
            raise ValueError(f"Tracking method {tracking_method} is not supported. Only 'trackpy' is available.")
        if tracking_params is None:
            tracking_params = {}
        tl_data = []
        total_frames = len(shape_names)
        frame_to_shape = {}
        for frame_idx, shape_name in enumerate(shape_names):
            for bbox_id in self.storage.get(shape_name, {}).get('displayed_ids', []):
                bbox = json.loads(self.storage[shape_name]['detection_data'][bbox_id]['bbox'])
                y1, x1, y2, x2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                tl_data.append({
                    'frame': frame_idx,
                    'x': cx,
                    'y': cy,
                    'bbox_id': bbox_id,
                    'score': self.storage[shape_name]['detection_data'][bbox_id].get('score'),
                })
            frame_to_shape[frame_idx] = shape_name

        if not tl_data:
            return
        df = pd.DataFrame(tl_data)
        search_range = tracking_params['search_range']
        memory = tracking_params['memory']
        create_missing_detections = tracking_params.get('create_missing_detections', False)
        tracked = tp.link_df(df, search_range=search_range, memory=memory)

        # Shift tracked particle ids to avoid conflicts with existing IDs
        id_shift_val = max([int(self.storage[shape_name]['next_id']) for shape_name in shape_names if shape_name in self.storage], default=0)
        tracked['particle'] = tracked['particle'] + id_shift_val

        for idx, row in tracked.iterrows():
            bbox_id = int(row['bbox_id'])
            frame_idx = int(row['frame'])
            shape_name = frame_to_shape[frame_idx]
            particle_id = int(row['particle'])
            print(f"{bbox_id} -> {particle_id} at frame {frame_idx} in shape {shape_name}")
            self.replace_detection_id(shape_name, bbox_id, particle_id)


        # Store frame idx for last encounter of each particle
        last_particle_layer_idx = {}
        for frame_idx in range(total_frames):
            shape_name = frame_to_shape[frame_idx]
            frame_rows = tracked[tracked['frame'] == frame_idx]

            for idx, row in frame_rows.iterrows():
                particle_id = int(row['particle'])
                last_particle_layer_idx[particle_id] = frame_idx

            # Remove old particles (> memory frames ago)
            last_particle_layer_idx = {particle_id: layer_idx for particle_id, layer_idx in last_particle_layer_idx.items() if layer_idx + memory >= frame_idx}
            
            cur_shape_name = frame_to_shape[frame_idx]
            if not cur_shape_name in self.storage:
                self.storage[cur_shape_name] = {
                    'detection_data': {},
                    'segmentation_data': {},
                    'image_size': imgs[frame_idx].shape[:2],
                    'displayed_ids': [],
                    'next_id': 1
                }

            for particle_id, prev_frame_idx in last_particle_layer_idx.items():
                if not particle_id in self.storage[cur_shape_name]['detection_data']:
                    # Only create missing detections if the checkbox is enabled
                    if create_missing_detections:
                        prev_shape_name = frame_to_shape[prev_frame_idx]
                        self.storage[cur_shape_name]['detection_data'][particle_id] = copy.deepcopy(self.storage[prev_shape_name]['detection_data'][particle_id])
                        # Ensure segmentation_data exists for current shape
                        if 'segmentation_data' not in self.storage[cur_shape_name]:
                            self.storage[cur_shape_name]['segmentation_data'] = {}
                        if particle_id in self.storage[prev_shape_name].get('segmentation_data', {}):
                            self.storage[cur_shape_name]['segmentation_data'][particle_id] = copy.deepcopy(self.storage[prev_shape_name]['segmentation_data'][particle_id])
                        self.storage[cur_shape_name]['next_id'] = max(self.storage[cur_shape_name]['next_id'], particle_id + 1)
            self._fill_default_data(cur_shape_name)