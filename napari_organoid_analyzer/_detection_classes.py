

import json
import uuid
from . import _utils


class DetectionLayer:
    """Abstract representation of a group of detected organoid, beloning to one image layer / frame.
    
    This class can be used to update the bbox and segmentation visualizations and to get features.
    """
    def __init__(self, image_layer, bbox_layer, segmentation_layer):
        self.image_layer = image_layer
        self.bbox_layer = bbox_layer
        self.segmentation_layer = segmentation_layer
        self.detections = []
        self.is_active = []  # True if the corresponding detection should be visible.

        self._score_thres = 0.5
        self._min_diameter = 30  # um

    def add_detection(self, detection: "DetectionInstance"):
        # Links the detection object to this layer
        self.detections.append(detection)
        # Links this layer to the detection object
        detection.set_layer_and_index(layer=self, idx=len(self.detections) - 1)

    def delete_detection(self, idx):
        assert idx < len(self.detections), (idx, len(self.detections))
        assert idx >= 0, idx

        # Removes the link to the detection
        self.detections.pop(idx)

        # Decreases the idx by 1 for all detection objects with higher idx.
        for i in range(idx+1, len(self.detections)):
            self.detections[i].set_layer_and_index(layer=self, idx=i-1)

    def _update_is_active(self):
        self.is_active = [
            (d.confidence >= self._score_thres) and (d.min_diameter > self._min_diameter) 
            for d in self.detections
        ]

    def set_score_thres(self, thres):
        self._score_thres = thres
        self._update_is_active()
        self.update_bbox_layer()

    def set_min_diameter(self, diameter):
        self._min_diameter = diameter
        self._update_is_active()
        self.update_bbox_layer()

    def update_bbox_layer(self):
        """Updates the visible bounding boxes in the shapes layer"""
        pass

    def get_bbox_layer_params(self):
        bboxes = [
            d.bbox_coordinates for d, is_active in zip(self.detections, self.is_active) if is_active 
        ]
        properties = [
            d.properties for d, is_active in zip(self.detections, self.is_active) if is_active 
        ]

        # Convert properties into dict[list]
        if len(properties) == 0:
            properties_out = dict()
        else:
            keys = properties[0].keys
            properties_out = {k: [] for k in keys}
            for p in properties:
                for k in keys:
                    properties_out[k].append(p[k])
            
        bboxes = _utils.convert_boxes_to_napari_view(bboxes)

        return bboxes, properties_out


class DetectionInstance:
    """Abstract representation of a single detected organoid.
    
    This class can be used to access and store features of the organoid.
    """
    def __init__(self, 
                 bbox_coordinates: tuple[tuple, tuple],
                 scale: tuple,
                 confidence: float = 1.0,
                 properties: dict | None = None):
        self.bbox_coordinates = bbox_coordinates
        self.scale = scale
        self.confidence = confidence
        self.properties = properties

        self.layer = None
        self.layer_idx = None
        self.id = uuid.uuid4()

        self.min_diameter = self._get_min_diameter()

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if not isinstance(other, DetectionInstance):
            # Only equality tests to other `DetectionInstance` instances are supported
            return NotImplemented
        return self.id == other.id
    
    def _get_min_diameter(self):
        assert len(self.scale) == 2, self.scale
        dx, dy = _utils.get_diams(self.bbox_coordinates)
        dx = dx / self.scale[0]
        dy = dy / self.scale[1]
        return min(dx, dy)

    def set_layer_and_index(self, layer: "DetectionLayer", idx: int):
        self.layer = layer
        self.layer_idx = idx

    
# Factory function

def create_detections_from_params(detection_data):
    """Creates a list of detection instances according to detection_data
    """
    detections = []
    for bbox_id, values in detection_data.items():
        bbox = json.loads(values['bbox'])
        values.pop('bbox')
        score = values.pop('score')
        
        values['bbox_id'] = int(bbox_id)  # TODO: replace with actual ID

        detections.append(
            DetectionInstance(bbox, 
                              scale=(1.0, 1.0), 
                              confidence=score,
                              properties=values)
        )
    
    return detections





# class DetectionTracker:
#     """Connects detected organoids as timelapse tracks.
#     """
#     def __init__(self, timelapse_name):
#         self.timelapse_name = timelapse_name
#         self.tracks = []

#     def create_new_track(self):
#         self.tracks.append(set())

#     def add_detection_to_track(self, detection: "DetectionInstance", track_id: int):
#         assert track_id < len(self.tracks), (track_id, len(self.tracks))
#         assert track_id >= 0, track_id

#         # Links the detection object to this track
#         self.tracks[track_id].add(detection)
#         # Links this layer to the detection object
#         detection.set_layer_and_index(layer=self, idx=len(self.detections) - 1)

#     def delete_detection(self, idx):
#         assert idx < len(self.detections), (idx, len(self.detections))

#         # Removes the link to the detection
#         self.detections.pop(idx)

#         # Decreases the idx by 1 for all detection objects with higher idx.
#         for i in range(idx+1, len(self.detections)):
#             self.detections[i].set_layer_and_index(layer=self, idx=i-1)

# class TrackingID:
#     _id_tracker = dict()
    
#     def __init__(self, timelapse_name):
#         if timelapse_name not in self._id_tracker:
#             self._id_tracker[timelapse_name] = 0
#         else:
#             self._id_tracker[timelapse_name] += 0

#         self.tracking_id = len(self._id_tracker[timelapse_name])
#         self.timelapse_name = timelapse_name