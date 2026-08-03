from pathlib import Path

def init():
    global MODELS
    MODELS = {
        "MultiOrg FRCNN": {
            "filename": "multiorg_frcnn.pth", 
            "source": "https://huggingface.co/marr-peng-lab/organoid_detection/resolve/main/multiorg_frcnn.pth"
        },
        "MultiOrg SSD": {
            "filename": "multiorg_ssd.pth", 
            "source": "https://huggingface.co/marr-peng-lab/organoid_detection/resolve/main/multiorg_ssd.pth"
        },
        "MultiOrg YOLOv3": {
            "filename": "multiorg_yolov3.pth", 
            "source": "https://huggingface.co/marr-peng-lab/organoid_detection/resolve/main/multiorg_yolov3.pth"
        },
        "MultiOrg RTMDet": {
            "filename": "multiorg_rtmdet.pth", 
            "source": "https://huggingface.co/marr-peng-lab/organoid_detection/resolve/main/multiorg_rtmdet.pth"
        },
        "SAMOS": {
            "filename": "SAMOS.pth", 
            "source": "https://huggingface.co/marr-peng-lab/organoid_detection/resolve/main/SAMOS.pth"
        },
        "FRCNN": {
            "filename": "FRCNN_general.pth", 
            "source": "https://huggingface.co/marr-peng-lab/organoid_detection/resolve/main/FRCNN_general.pth"
        },
    }

    global SAM_MODEL
    SAM_MODEL = {"filename": "sam_vit_l_0b3195.pth",
                 "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"}
    
    global CACHE_DIR
    CACHE_DIR = Path.home() / ".cache/napari-organoid-analyzer"
    
    global SETTINGS_DIR
    SETTINGS_DIR = CACHE_DIR / "settings"
    
    global MODELS_DIR
    MODELS_DIR = CACHE_DIR / "models"

    global DETECTIONS_DIR
    DETECTIONS_DIR = CACHE_DIR / "detections-cache"

    global ARCHITECTURES_DIR
    ARCHITECTURES_DIR = CACHE_DIR / "architectures"

    global TRAINED_MODELS_DIR
    TRAINED_MODELS_DIR = CACHE_DIR / "trained_models"

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHITECTURES_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    global MODEL_TYPE
    MODEL_TYPE = '.pth'
    
    # Add color definitions
    global COLOR_CLASS_1
    COLOR_CLASS_1 = [85 / 255, 1.0, 0, 1.0]  # Green
    
    global COLOR_CLASS_2
    COLOR_CLASS_2 = [0, 29 / 255, 1.0, 1.0]  # Blue

    global COLOR_DEFAULT
    COLOR_DEFAULT = [1.0, 0, 0, 1.0]  # Red

    global TEXT_COLOR
    TEXT_COLOR = [1.0, 0, 0, 1.0]  # Red for text labels

    global DENSITY_K_NEIGHBORS
    DENSITY_K_NEIGHBORS = 2

