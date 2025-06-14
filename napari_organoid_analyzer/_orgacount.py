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
from napari_organoid_analyzer._utils import set_posix_windows
import matplotlib.pyplot as plt
import cv2
import sys
import logging
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_SAMOS'))

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
        self.pred_bboxes = {}
        self.pred_scores = {}
        self.pred_masks = {}
        self.signal_masks = {}
        self.pred_ids = {}
        self.next_id = {}
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
        self.pred_bboxes[shapes_name] = bboxes
        self.pred_scores[shapes_name] = pred_scores
        num_predictions = bboxes.size(0)
        self.pred_ids[shapes_name] = [int(i+1) for i in range(num_predictions)]
        self.next_id[shapes_name] = num_predictions+1

    def run_segmentation(self, img, mask_name, bboxes, signal_fields):
        """
        Runs segmentation pipeline for selected image, based on previously detected bboxes
        Inputs
        ----------
        img: Numpy array of size [H, W, 3]
            The input image
        mask_name: str
            Name of mask
        bbooxes: Numpy array of size [N, 4]
            Array of all predicted bboxes in xyxy format
        signal_field: dict({signal_name: signal_field})
            Optional signal fields for the image
        """
        signal_masks = {}
        if bboxes.shape[0] == 0:
            pred_mask = np.array([])
        else:
            img = _utils.normalize(img)
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
                signal_masks[signal_name] = signal_mask
            
        self.pred_masks[mask_name] = pred_mask
            
        if len(signal_masks) > 0:
            self.signal_masks[mask_name] = signal_masks
            assert len(pred_mask) == len(signal_mask)
        features = []
        for idx in range(len(pred_mask)):
            features.append(self._compute_features(
                pred_mask[idx], 
                idx, 
                {signal_name: (signal_mask[idx], signal_fields[signal_name][:, :, 0]) for signal_name, signal_mask in signal_masks.items()}, 
            ))
        features = {key: [feature[key] for feature in features] for key in features[0]}
        return pred_mask, features, signal_masks
    
    def _compute_features(self, mask, idx, signal_data):
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
            logging.warning(f"Multiple contours found for organoid {idx}. Using the contour with largest area: {area[id_selected]}")
        perimeter = cv2.arcLength(contours[id_selected], True)
        if perimeter > 0:
            roundness = (4 * np.pi * area[id_selected]) / (perimeter ** 2)
        else:
            roundness = 0
        features = {
            'Area (pixel units)': area[id_selected],
            'Perimeter (pixel units)': perimeter,
            'Roundness': roundness
        }
        for signal_name, (signal_mask, signal_field) in signal_data.items():
            signal_contours, _ = cv2.findContours(signal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_signal_area = sum([cv2.contourArea(contour) for contour in signal_contours])
            signal_total_intensity = np.sum(signal_mask * signal_field) / 255.0
            signal_mean_intensity = signal_total_intensity / total_signal_area
            features.update({
                f"{signal_name} area (pixel units)": total_signal_area,
                f"{signal_name} mean intensity": signal_mean_intensity,
                f"{signal_name} total intensity": signal_total_intensity,
                f"{signal_name} components": len(signal_contours)
            })
        return features


    def apply_params(self, shapes_name, confidence, min_diameter_um):
        """ After results have been stored in dict this function will filter the dicts based on the confidence
        and min_diameter_um thresholds for the given results defined by shape_name and return the filtered dicts. """
        pred_bboxes, pred_scores, pred_ids = self._apply_confidence_thresh(shapes_name, confidence)
        if pred_bboxes.size(0)!=0:
            pred_bboxes, pred_scores, pred_ids = self._filter_small_organoids(pred_bboxes, pred_scores, pred_ids, min_diameter_um)
        pred_bboxes = _utils.convert_boxes_to_napari_view(pred_bboxes)
        return pred_bboxes, pred_scores, pred_ids

    def _apply_confidence_thresh(self, shapes_name, confidence):
        """ Filters out results of shapes_name based on the current confidence threshold. """
        if shapes_name not in self.pred_bboxes.keys(): return torch.empty((0)), torch.empty((0)), []
        keep = (self.pred_scores[shapes_name]>confidence).nonzero(as_tuple=True)[0]
        if len(keep) == 0: return torch.empty((0)), torch.empty((0)), []
        result_bboxes = self.pred_bboxes[shapes_name][keep]
        result_scores = self.pred_scores[shapes_name][keep]
        result_ids = [self.pred_ids[shapes_name][int(i)] for i in keep.tolist()]
        return result_bboxes, result_scores, result_ids
    
    def _filter_small_organoids(self, pred_bboxes, pred_scores, pred_ids, min_diameter):
        """ Filters out small result boxes of shapes_name based on the current min diameter size. """
        if len(pred_bboxes)==0: return torch.empty((0)), torch.empty((0)), []
        min_diameter_x = min_diameter / self.img_scale[0]
        min_diameter_y = min_diameter / self.img_scale[1]
        keep = []
        for idx in range(len(pred_bboxes)):
            dx, dy = _utils.get_diams(pred_bboxes[idx])
            if (dx >= min_diameter_x and dy >= min_diameter_y) or pred_scores[idx] == 1: keep.append(idx)
        if len(keep) == 0: return torch.empty((0)), torch.empty((0)), []
        pred_bboxes = pred_bboxes[keep]
        pred_scores = pred_scores[keep]
        pred_ids = [pred_ids[i] for i in keep]
        return pred_bboxes, pred_scores, pred_ids

    def update_bboxes_scores(self, shapes_name, new_bboxes, new_scores, new_ids, old_confidence, old_min_diameter):
        ''' Updated the results dicts, self.pred_bboxes, self.pred_scores and self.pred_ids with new results.
        If the shapes name doesn't exist as a key in the dicts the results are added with the new key. If the
        key exists then new_bboxes, new_scores and new_ids are compared to the class result dicts and the dicts 
        are updated, either by adding some box (user added box) or removing some box (user deleted a prediction).'''
        new_bboxes = _utils.convert_boxes_from_napari_view(new_bboxes)
        new_scores =  torch.Tensor(list(new_scores))
        new_ids = list(new_ids)
        # if run hasn't been run
        if shapes_name not in self.pred_bboxes.keys():
            self.pred_bboxes[shapes_name] = new_bboxes
            self.pred_scores[shapes_name] = new_scores
            self.pred_ids[shapes_name] = new_ids
            self.next_id[shapes_name] = len(new_ids)+1

        elif len(new_ids)==0: return

        else:
            min_diameter_x = old_min_diameter / self.img_scale[0]
            min_diameter_y = old_min_diameter / self.img_scale[1]
            # find ids that are not in self.pred_ids but are in new_ids
            added_box_ids = list(set(new_ids).difference(self.pred_ids[shapes_name]))
            if len(added_box_ids) > 0:
                added_ids = [new_ids.index(box_id) for box_id in added_box_ids]
                #  and add them
                self.pred_bboxes[shapes_name] = torch.cat((self.pred_bboxes[shapes_name], new_bboxes[added_ids]))
                self.pred_scores[shapes_name] = torch.cat((self.pred_scores[shapes_name], new_scores[added_ids]))
                new_ids_to_add = [new_ids[i] for i in added_ids]
                self.pred_ids[shapes_name].extend(new_ids_to_add)
            
            # and find ids that are in self.pred_ids and not in new_ids
            potential_removed_box_ids = list(set(self.pred_ids[shapes_name]).difference(new_ids))
            if len(potential_removed_box_ids) > 0:
                potential_removed_ids = [self.pred_ids[shapes_name].index(box_id) for box_id in potential_removed_box_ids]
                remove_ids = []
                for idx in potential_removed_ids:
                    dx, dy  = _utils.get_diams(self.pred_bboxes[shapes_name][idx])
                    if self.pred_scores[shapes_name][idx] > old_confidence and dx > min_diameter_x and dy > min_diameter_y:
                        remove_ids.append(idx)
                # and remove them
                for idx in reversed(remove_ids):
                    self.pred_bboxes[shapes_name] = torch.cat((self.pred_bboxes[shapes_name][:idx, :], self.pred_bboxes[shapes_name][idx+1:, :]))
                    self.pred_scores[shapes_name] = torch.cat((self.pred_scores[shapes_name][:idx], self.pred_scores[shapes_name][idx+1:]))
                    new_pred_ids = self.pred_ids[shapes_name][:idx]
                    new_pred_ids.extend(self.pred_ids[shapes_name][idx+1:])
                    self.pred_ids[shapes_name] = new_pred_ids
            self.next_id[shapes_name] = max(self.pred_ids[shapes_name]) + 1

    def remove_shape_from_dict(self, shapes_name):
        """ Removes results of shapes_name from all result dicts. """
        self.pred_bboxes.pop(shapes_name, None)
        self.pred_scores.pop(shapes_name, None)
        self.pred_ids.pop(shapes_name, None)
        self.next_id.pop(shapes_name, None)
        self.pred_masks.pop(shapes_name, None)