from pathlib import Path

def init():
    
    global MODELS
    MODELS = {
        "faster r-cnn": {"filename": "faster-rcnn_r50_fpn_organoid_best_coco_bbox_mAP_epoch_68.pth", 
                         "source": "https://zenodo.org/records/11388549/files/faster-rcnn_r50_fpn_organoid_best_coco_bbox_mAP_epoch_68.pth"
                         },
        "ssd": {"filename": "ssd_organoid_best_coco_bbox_mAP_epoch_86.pth", 
                "source": "https://zenodo.org/records/11388549/files/ssd_organoid_best_coco_bbox_mAP_epoch_86.pth"
                },
        "yolov3": {"filename": "yolov3_416_organoid_best_coco_bbox_mAP_epoch_27.pth",
                   "source": "https://zenodo.org/records/11388549/files/yolov3_416_organoid_best_coco_bbox_mAP_epoch_27.pth"
                   },
        "rtmdet":  {"filename": "rtmdet_l_organoid_best_coco_bbox_mAP_epoch_323.pth",
                    "source": "https://zenodo.org/records/11388549/files/rtmdet_l_organoid_best_coco_bbox_mAP_epoch_323.pth"
                    },
        "SAMOS": {"filename": "own_checkpoint_last.ckpt", 
                  "source": "https://huggingface.co/marr-peng-lab/organoid_detection/resolve/main/own_checkpoint_last.ckpt"},
    }

    global SAM_MODEL
    SAM_MODEL = {"filename": "sam_vit_l_0b3195.pth",
                           "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"}
    
    global CACHE_DIR
    CACHE_DIR = Path.home() / ".cache/napari-organoid-analyzer"
    
    global MODELS_DIR
    MODELS_DIR = CACHE_DIR / "models"

    global DETECTIONS_DIR
    DETECTIONS_DIR = CACHE_DIR / "detections-cache"

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    global MODEL_TYPE
    MODEL_TYPE = '.pth'

    global CONFIGS
    CONFIGS = {
        "faster r-cnn": {"source": "https://zenodo.org/records/11388549/files/faster-rcnn_r50_fpn_organoid.py",
                        "destination": ".mim/configs/faster_rcnn/faster-rcnn_r50_fpn_organoid.py"
                        },
        "ssd": {"source": "https://zenodo.org/records/11388549/files/ssd_organoid.py",
                "destination": ".mim/configs/ssd/ssd_organoid.py"
                },
        "yolov3": {"source": "https://zenodo.org/records/11388549/files/yolov3_416_organoid.py",
                "destination": ".mim/configs/yolo/yolov3_416_organoid.py"
                },
        "rtmdet":  {"source": "https://zenodo.org/records/11388549/files/rtmdet_l_organoid.py",
                    "destination": ".mim/configs/rtmdet/rtmdet_l_organoid.py"
                    }
        # No config needed for SAMOS

}
    
    # Add color definitions
    global COLOR_CLASS_1
    COLOR_CLASS_1 = [85 / 255, 1.0, 0, 1.0]  # Green
    
    global COLOR_CLASS_2
    COLOR_CLASS_2 = [0, 29 / 255, 1.0, 1.0]  # Blue

    global COLOR_DEFAULT
    COLOR_DEFAULT = [1.0, 0, 0, 1.0]  # Red

    global TEXT_COLOR
    TEXT_COLOR = [1.0, 0, 0, 1.0]  # Red for text labels

