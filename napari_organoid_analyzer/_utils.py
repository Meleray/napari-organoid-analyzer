from contextlib import contextmanager
import os
import pathlib
from pathlib import Path
import pkgutil

import numpy as np
import math
import json
import csv
from skimage.transform import rescale
from skimage.color import gray2rgb
from skimage.draw import polygon as skimage_polygon
import hashlib
import cv2

import torch
from torchvision.ops import nms

from napari_organoid_analyzer import settings

from ctypes.wintypes import MAX_PATH


def normalize(img, grayscale=True, correct_bg=True, fix_median=True):
    """Normalizes and corrects the background of a single image and converts it to 3-channel grayscale.
    """
    if img.ndim==2:
        img = np.stack([img, img, img], axis=2)
    elif img.ndim==3:
        if img.shape[2]==1:
            img = np.concat([img, img, img], axis=2)
        elif img.shape[2]==4:
            img = img[:, :, :3]
        elif img.shape[2]==3:
            pass
        else:
            raise RuntimeError(img.shape)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack([img, img, img], axis=2)
    else:
        raise RuntimeError(img.shape)
    
    # Correct background
    if correct_bg:
        pass
        # bg = ...

    if fix_median:
        pass
    return img


@contextmanager
def set_posix_windows():
    """Replaces PosixPath temporarily with WindowsPath on Windows.
    
    Necessary for loading model checkpoints which contain PosixPaths on Windows."""
    posix_backup = pathlib.PosixPath
    try:
        if os.name == 'nt': # Only on Windows, replace PosixPath temporarily with WindowsPath
            pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def check_filename_integrity(path: Path):
    if os.name == 'nt':
        path_max = MAX_PATH
    else:
        try:
            path_max = os.pathconf(path.parent, 'PC_PATH_MAX')
        except AttributeError:
            # Set fixed to 4096
            path_max = 4096
    
    if len(str(path)) >= path_max:
        raise RuntimeWarning(f'The file path exceeds the maximum length of {path_max} letters by {len(str(path)) + 1 - path_max}. Please choose a filename or folder with a shorter path.')


def collate_instance_masks(masks, color=False):
    """
    Merges instance-based masks into a single RGB image where each instance has a random color.
    Args:
        masks np.ndarray: List of binary masks for each instance of dimension [N, H, W]
    
    Returns:
        np.ndarray: RGB image of dimension [H, W, 3] where each instance has a unique random color
    """
    if not color:
        return np.any(masks, axis=0)
    
    num_instances, height, width = masks.shape
    result = np.zeros((height, width, 3), dtype=np.uint8)
    colors = np.random.randint(0, 256, (num_instances, 3))
    for i in range(num_instances):
        for c in range(3):
            result[:, :, c] = np.where(masks[i], colors[i, c], result[:, :, c])
    return result

def add_local_models():
    """ Checks the models directory for any local models previously added by the user.
    If some are found then these are added to the model dictionary (see settings). """
    if not os.path.exists(settings.MODELS_DIR): return
    model_names_in_dir = [file for file in os.listdir(settings.MODELS_DIR)]
    model_names_in_dict = [settings.MODELS[key]["filename"] for key in settings.MODELS.keys()]
    for model_name in model_names_in_dir:
        if model_name not in model_names_in_dict and model_name.endswith(settings.MODEL_TYPE) and model_name != settings.SAM_MODEL["filename"]:
            _ = add_to_dict(model_name)

def add_to_dict(filepath):
    """ Given the full path and name of a model in filepath the model is added to the models dict (see settings)"""
    filepath = Path(filepath)
    name = filepath.name
    stem_name = filepath.stem
    settings.MODELS[stem_name] = {"filename": name, "source": "local"}
    return stem_name

def return_is_file(path, filename):
    """ Return True if the file exists in path and False otherwise """
    full_path = join_paths(path, filename)
    return os.path.isfile(full_path)

def join_paths(path1, path2):
    """ Returns output of os.path.join """
    return os.path.join(path1, path2)

@contextmanager
def set_dict_key(dictionary, key, value):
    """ Used to set a new value in the napari layer metadata """
    dictionary[key] = value
    yield
    del dictionary[key]

def get_diams(bbox):
    """ Get the lengths of the bounding boxes """
    x1_real, y1_real, x2_real, y2_real = bbox
    dx = abs(x1_real - x2_real)
    dy = abs(y1_real - y2_real)
    return dx, dy

def write_to_json(name: Path, data):
    """ Write data to a json file. Here data is a dict """
    check_filename_integrity(name)
    with open(name, 'w') as outfile:
        json.dump(data, outfile)  

def get_bboxes_as_dict(bboxes, bbox_ids, scores, scales):
    """ Write all data, boxes, ids and scores, scale and class label, to a dict so we can later save as a json """
    data_json = {} 
    for idx, bbox in enumerate(bboxes):
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]

        data_json.update({str(bbox_ids[idx]): {'box_id': str(bbox_ids[idx]),
                                                'x1': str(x1),
                                                'x2': str(x2),
                                                'y1': str(y1),
                                                'y2': str(y2),
                                                'confidence': str(scores[idx]),
                                                'scale_x': str(scales[0]),
                                                'scale_y': str(scales[1]),
                                                }
                        })
    return data_json

def compute_image_hash(image):
    """Compute a hash of the image for caching purposes"""
        # Convert image to bytes and calculate MD5 hash
    image_bytes = np.array(image).tobytes()
    image_hash = hashlib.md5(image_bytes).hexdigest()
    return image_hash

def write_to_csv(name: Path, data):
    """ Write data to a csv file. Here data is a list of lists, where each item represents a row in the csv file. """
    check_filename_integrity(name)
    with open(name, 'w') as f:
        write = csv.writer(f, delimiter=';')
        write.writerow(['OrganoidID', 'D1[um]','D2[um]', 'Area [um^2]'])
        write.writerows(data)

def get_bbox_diameters(bboxes, bbox_ids, scales):
    """ Write all data, box diameters and area, ids and scale, to a list so we can later save as a csv """
    data_csv = []
    # save diameters and area of organoids (approximated as ellipses)
    for idx, bbox in enumerate(bboxes):
        d1 = abs(bbox[0][0] - bbox[2][0]) * scales[0]
        d2 = abs(bbox[0][1] - bbox[2][1]) * scales[1]
        area = math.pi * d1 * d2
        data_csv.append([bbox_ids[idx], round(d1,3), round(d2,3), round(area,3)])
    return data_csv

def squeeze_img(img):
    """ Squeeze image - all dims that have size one will be removed """
    return np.squeeze(img)

def prepare_img(test_img, step, window_size, rescale_factor):
    """ The original image is prepared for running model inference """
    # squeeze and resize image
    test_img = squeeze_img(test_img)
    test_img = rescale(test_img, rescale_factor, preserve_range=True)
    img_height, img_width = test_img.shape
    # pad image
    pad_x = (img_height//step)*step + window_size - img_height
    pad_y = (img_width//step)*step + window_size - img_width
    test_img = np.pad(test_img, ((0, int(pad_x)), (0, int(pad_y))), mode='edge')
    # normalise and convert to RGB - model input has size 3
    test_img = (test_img-np.min(test_img))/(np.max(test_img)-np.min(test_img)) 
    test_img = (255*test_img).astype(np.uint8)
    test_img = gray2rgb(test_img) #[H,W,C]

    # convert from RGB to GBR - expected from DetInferencer 
    test_img = test_img[..., ::-1] 
    
    return test_img, img_height, img_width

def apply_nms(bbox_preds, scores_preds, iou_thresh=0.5):
    """ Function applies non max suppression to iteratively remove lower scoring boxes which have an IoU greater than iou_threshold 
    with another (higher scoring) box. The boxes and corresponding scores whihc remain are returned. """
    # torchvision returns the indices of the bboxes to keep
    keep = nms(bbox_preds, scores_preds, iou_thresh)
    # filter existing boxes and scores and return
    bbox_preds_kept = bbox_preds[keep]
    scores_preds = scores_preds[keep]
    return bbox_preds_kept, scores_preds

def convert_boxes_to_napari_view(pred_bboxes):
    """ The bboxes are converted from tensors in model output form to a form which can be visualised in the napari viewer """
    if pred_bboxes is None: return []
    new_boxes = []
    for idx in range(len(pred_bboxes)):
        # convert to numpy and take coordinates 
        x1_real, y1_real, x2_real, y2_real = pred_bboxes[idx]
        # append to a list in form napari exects
        new_boxes.append(np.array([[x1_real, y1_real],
                                [x1_real, y2_real],
                                [x2_real, y2_real],
                                [x2_real, y1_real]]))
    return new_boxes

def convert_boxes_from_napari_view(pred_bboxes):
    """ The bboxes are converted from the form they were in the napari viewer to tensors that correspond to the model output form """
    new_boxes = []
    for idx in range(len(pred_bboxes)):
        # read coordinates
        x1 = pred_bboxes[idx][0][0]
        x2 = pred_bboxes[idx][2][0]
        y1 = pred_bboxes[idx][0][1]
        y2 = pred_bboxes[idx][2][1]
        # convert to tensor and append to list
        new_boxes.append(torch.Tensor([x1, y1, x2, y2]))
    if len(new_boxes) > 0: new_boxes = torch.stack(new_boxes)
    return new_boxes.tolist() if len(new_boxes) > 0 else []

def apply_normalization(img):
    """ Normalize image"""
    # squeeze and change dtype
    img = squeeze_img(img)
    img = img.astype(np.float64)
    # adapt img to range 0-255
    if img.ndim == 3:
        img_min = np.min(img) # 31.3125 png 0
        img_max = np.max(img) # 2899.25 png 178
        img_norm = (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)
    elif img.ndim == 4:
        img_norm = np.zeros_like(img, dtype=np.uint8)
        for idx in range(len(img)):
            frame = img[idx]
            frame_min = np.min(frame)
            frame_max = np.max(frame)
            frame_norm = (255 * (frame - frame_min) / (frame_max - frame_min)).astype(np.uint8)
            img_norm[idx] = frame_norm
    elif img.ndim == 2:
        img_min = np.min(img)
        img_max = np.max(img)
        img_norm = (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)
    else:
        raise ValueError(f"Wrong image format for preprocessing. Image shape: {img.shape}. Please delete the added image.")
    return img_norm

def get_package_init_file(package_name):
    loader = pkgutil.get_loader(package_name)
    if loader is None or not hasattr(loader, 'get_filename'):
        raise ImportError(f"Cannot find package {package_name}")
    package_path = loader.get_filename(package_name)
    # Determine the path to the __init__.py file
    if os.path.isdir(package_path):
        init_file_path = os.path.join(package_path, '__init__.py')
    else:
        init_file_path = package_path
    if not os.path.isfile(init_file_path):
        raise FileNotFoundError(f"__init__.py file not found for package {package_name}")
    return init_file_path

def update_version_in_mmdet_init_file(package_name, old_version, new_version):
    init_file_path = get_package_init_file(package_name)
    with open(init_file_path, 'r') as file:
        lines = file.readlines()
    with open(init_file_path, 'w') as file:
        for line in lines:
            if f"mmcv_maximum_version = '{old_version}'" in line:
                file.write(line.replace(old_version, new_version))

def validate_bboxes(bboxes, image_shape):
    """
    Validates whether all bounding boxes are contained within the image dimensions.

    Args:
        bboxes (list of np.ndarray): List of bounding boxes, where each box
                                     is a 4x2 numpy array of its vertices
                                     (e.g., [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]).
        image_shape (tuple): Shape of the image as (height, width).

    Returns:
        bool: True if all bounding boxes are valid, False otherwise.
    """
    img_height, img_width = image_shape
    for bbox_vertices in bboxes:
        x1 = bbox_vertices[0][0]
        x2 = bbox_vertices[2][0]
        y1 = bbox_vertices[0][1]
        y2 = bbox_vertices[2][1]
        
        if not (0 <= x1 < x2 <= img_height and 0 <= y1 < y2 <= img_width):
            return False
    return True

def get_timelapse_name(name):
    """ Get the name of the timelapse from the napari layer name """
    return '_'.join(name.split('_')[2:])

def mask2polygon(mask):
    """
    Convert a binary mask to a polygon (sequence of vertices).
    Returns a list of (x, y) tuples for the largest contour.
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.squeeze()
    if polygon.ndim == 1:
        polygon = polygon[np.newaxis, :]
    return polygon.tolist()

def polygon2mask(polygon, shape):
    """
    Convert a polygon (sequence of vertices) and image shape to a binary mask.
    polygon: list or array of (x, y) or [ [x1, y1], [x2, y2], ... ]
    shape: (height, width)
    Returns a binary mask of the given shape.
    """
    polygon = np.array(polygon)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("Polygon must be a sequence of (x, y) points.")
    rr, cc = skimage_polygon(polygon[:, 1], polygon[:, 0], shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rr, cc] = 1
    return mask