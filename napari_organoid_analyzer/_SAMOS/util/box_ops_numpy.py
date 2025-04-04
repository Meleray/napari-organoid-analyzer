import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def mask_to_boxes(labels):
    boxes = []
    for i in sorted(np.unique(labels).astype(int).tolist()):
        if i == 0:  # background
            continue
        mask = labels == i
        y_coords, x_coords = np.nonzero(mask)
        box = [y_coords.min(), x_coords.min(), y_coords.max(), x_coords.max()]
        boxes.append(box)
    return np.array(boxes).reshape((-1, 4))


def cxcywh_to_xyxy(boxes):
    # Convert the boxes from center format to [x_min, y_min, x_max, y_max] format
    converted_boxes = np.zeros_like(boxes)
    converted_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0  # x_min
    converted_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0  # y_min
    converted_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0  # x_max
    converted_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0  # y_max
    return converted_boxes

def xyxy_to_cxcywh(boxes):
    # Convert the boxes from [x_min, y_min, x_max, y_max] format to center format 
    converted_boxes = np.zeros_like(boxes)
    converted_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0  # center x
    converted_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0  # center y
    converted_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    converted_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return converted_boxes

def plot_boxes(image, boxes, format='cycxhw_01', color='red', ax=None, show_image=True, linewidth=2):
    """`boxes` should be cycxhw format normalized by the longest image side to values in [0, 1].
    """
    H, W = image.shape[:2]

    if format=='cycxhw_01':
        # boxes are in cycxhw format normalized by the longest image side. Converts to min/max coordinates in pixels.
        original_boxes = cxcywh_to_xyxy(boxes) * max(H, W)
    elif format=='cycxhw_px':
        # boxes are in cycxhw format in pixels. Converts to min/max coordinates in pixels.
        original_boxes = cxcywh_to_xyxy(boxes)
    elif format=='yxyx_01':
        # boxes are in min/max coordinates normalized by the longest image side. Converts to min/max coordinates in pixels.
        original_boxes = boxes * max(H, W)
    elif format=='yxyx_px':
        # boxes are in min/max coordinates in pixels. No conversion necessary.
        original_boxes = boxes
    elif format=='xyxy_px':
        # boxes are in min/max coordinates in pixels. No conversion necessary.
        original_boxes = np.stack([
            boxes[:, 1],
            boxes[:, 0],
            boxes[:, 3],
            boxes[:, 2],
        ], axis=1)
    else:
        raise ValueError(format)

    # Plot the image
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8, 8), dpi=120)
    if show_image:
        ax.imshow(image)

    # Plot each bounding box
    for box in original_boxes:
        # The coordinates of the box are in [y_min, x_min, y_max, x_max] format
        y_min, x_min, y_max, x_max = box
        
        # Calculate width and height of the box
        width = x_max - x_min
        height = y_max - y_min
        
        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=linewidth, edgecolor=color, facecolor='none', alpha=0.7)
        
        # Add the patch to the Axes
        ax.add_patch(rect)

    # Display the plot
    # plt.show()
