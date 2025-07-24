
from copy import copy
import json
import uuid
import numpy as np
from . import _utils


class DetectionLayer:
    """Abstract representation of a group of detected organoid, beloning to one image layer / frame.
    
    This class can be used to update the bbox and segmentation visualizations and to get features.
    """
    def __init__(self, image_layer, bbox_layer, segmentation_layer):
        self.image_layer = image_layer
        self.bbox_layer = bbox_layer
        self.segmentation_layer = segmentation_layer
        self.scale=(1.0, 1.0)  # TODO: get scale from image_layer

        self.detections = dict()
        self.is_active = dict()  # True if the corresponding detection should be visible.

        self._score_thres = 0.5
        self._min_diameter = 30  # um

    def add_detection(self, detection: "DetectionInstance"):
        assert detection.get_id() not in self.detections, f"The detection object {detection.get_id()} is already added to this layer."
        # Links the detection object to this layer
        self.detections[detection.get_id()] = detection
        self.is_active[detection.get_id()] = self._is_active(detection)

        # detection.set_layer_and_index(layer=self, idx=len(self.detections) - 1)

        # Links this layer to the detection object
        detection.set_layer(layer=self)

    def delete_detection(self, id):
        assert id in self.detections, (id, self.detections.keys())

        # Removes the link to the detection
        self.detections.pop(id)
        self.is_active.pop(id)

        # # Decreases the idx by 1 for all detection objects with higher idx.
        # for i in range(idx+1, len(self.detections)):
        #     self.detections[i].set_layer_and_index(layer=self, idx=i-1)

    def _is_active(self, det: "DetectionInstance"):
        return (det.confidence >= self._score_thres) and (det.min_diameter > self._min_diameter)

    def _update_is_active(self):
        del self.is_active
        self.is_active = {
            id: self._is_active(det) for id, det in self.detections.items()
        }

        # self.is_active = [
        #     self._is_active(d) for d in self.detections
        # ]

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

    def update_active_detections(self, new_bboxes, new_properties):
        """Updates the detection data of active detections.
        
        The active detections should be 'new_bboxes'. 'new_properties[key]' should 
        either have the same length or be None.
        
        If one of the currently active detections is not present in new_bboxes, 
        this function deletes the detection.
        """
        # Update and add new detections
        new_active_ids = []
        print('update_active_detections')
        print('new_bboxes', new_bboxes)
        print('new_properties', new_properties)
        prop_keys = new_properties.keys()

        for i, bbox in enumerate(new_bboxes):
            # if len(bbox) != 2:
            #     print('Skipping shape', bbox, 'since it is not a bounding box')
            #     continue
            bbox = _utils.convert_napari_shape_to_bbox(bbox)
            prop = {k: new_properties[k][i] for k in prop_keys}
            print('bbox', bbox)
            print('prop', prop)

            if ('uuid' in prop) and (prop['uuid'] is not None):
                # Update data of existing detection
                self.detections[prop['uuid']].update_data(bbox, prop)

                new_active_ids.append(prop['uuid'])
            else:
                # Creates a new detection
                new_detection = DetectionInstance(
                    bbox_coordinates=bbox,
                    scale=self.scale,
                    confidence=1.0,
                    properties=prop
                )
                self.add_detection(new_detection)

                new_active_ids.append(new_detection.get_id())
        
        # Delete detections
        to_delete = []
        for id, is_active in self.is_active.items():
            if is_active and (id not in new_active_ids):
                to_delete.append(id)
        for id in to_delete:
            detection = self.detections[id]
            detection.delete()

    def get_bbox_layer_params(self):
        bboxes = []
        properties = []
        bbox_id = 0
        print('self.detections', self.detections)
        for id, detection in self.detections.items():
            if self.is_active[id]:
                bboxes.append(detection.bbox_coordinates)
                prop = detection.get_properties()
                prop['bbox_id'] = bbox_id
                properties.append(prop)
                bbox_id += 1
        # bboxes = [
        #     d.bbox_coordinates for d, is_active in zip(self.detections, self.is_active) if is_active 
        # ]
        # properties = [
        #     d.properties for d, is_active in zip(self.detections, self.is_active) if is_active 
        # ]

        # Convert properties into dict[list]
        if len(properties) == 0:
            properties_out = dict()
        else:
            keys = properties[0].keys()
            properties_out = {k: [] for k in keys}
            for p in properties:
                for k in keys:
                    properties_out[k].append(p[k])
            
        bboxes = _utils.convert_boxes_to_napari_view(bboxes)
        print('bboxes', bboxes)

        return bboxes, properties_out


class DetectionInstance:
    """Abstract representation of a single detected organoid.
    
    This class can be used to access and store features of the organoid.
    """
    def __init__(self, 
                 bbox_coordinates,
                 scale: tuple,
                 confidence: float = 1.0,
                 properties: dict | None = None):
        """Initializes a new detection object
        
        bbox_coordinates are expected to be a np.ndarray [xmin, ymin, xmax, ymax] in pixel coordinates.
        """
        self._set_bbox(bbox_coordinates)  # Unit: pixel
        self.scale = scale  # Unit: um / pixel
        self.confidence = confidence
        self.properties = properties

        self.layer = None
        # self.layer_idx = None

        self._id = uuid.uuid4()
        self._compute_bbox_features()

    def get_id(self):
        return self._id

    def __eq__(self, other):
        if not isinstance(other, DetectionInstance):
            # Only equality tests to other `DetectionInstance` instances are supported
            return NotImplemented
        return self.get_id() == other.get_id()
    
    def _set_bbox(self, bbox):
        assert len(bbox) == 4, bbox
        assert np.all(bbox[:2] <= bbox[2:]), bbox
        self.bbox_coordinates = bbox

    def _get_min_diameter(self):
        assert len(self.scale) == 2, self.scale
        dx, dy = _utils.get_diams(self.bbox_coordinates)
        dx = dx * self.scale[0]
        dy = dy * self.scale[1]
        return min(dx, dy)

    def set_layer(self, layer: "DetectionLayer"):
        self.layer = layer

    # def rerun_segmentation(self):
    #     """Reruns the segmentation step, if it depends on the bbox."""
    #     pass

    def update_data(self, bbox=None, properties=None):
        if properties is not None:
            self.properties.update(properties)

        bbox_changed = False
        if bbox is not None:
            if not np.allclose(self.bbox_coordinates, bbox, atol=2):
                # Shape changed more than 2 pixels by the user, set confidence to 1.0
                self.confidence = 1.0
                self._set_bbox(bbox)
                # self.rerun_segmentation()
                bbox_changed = True

        # Recompute features which depend on the bbox to avoid inconsistency if 
        # they are overwritten by properties or if the bbox changed.    
        if (properties is not None) or bbox_changed:
            self._compute_bbox_features()

    def _compute_bbox_features(self):
        self.min_diameter = self._get_min_diameter()
        xmin, ymin, xmax, ymax = self.bbox_coordinates
        self.set_property('bbox_area', abs(xmax - xmin) * abs(ymax-ymin) * self.scale[0] * self.scale[1])
    
    def set_property(self, key, value):
        if self.properties is None:
            self.properties = dict()
        
        if key in ['uuid', 'score']:
            return
        
        self.properties[key] = value

    def get_properties(self):
        if self.properties is None:
            self.properties = dict()

        prop = copy(self.properties)

        prop['score'] = self.confidence
        prop['uuid'] = self.get_id()
        return prop

    def delete(self):
        # Remove links to all other objects
        if self.layer is not None:
            self.layer.delete_detection(self.get_id())

    # def set_layer_and_index(self, layer: "DetectionLayer", idx: int):
    #     self.layer = layer
    #     self.layer_idx = idx

    
# Factory function

def create_detections_from_params(detection_data):
    """Creates a list of detection instances according to detection_data
    """
    detections = []
    for bbox_id, values in detection_data.items():
        bbox = json.loads(values['bbox'])
        bbox = _utils.convert_napari_shape_to_bbox(bbox)
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