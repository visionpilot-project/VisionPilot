from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"
VIDEOS_DIR = BASE_DIR / "videos"

VEHICLE_PEDESTRIAN_MODEL = MODELS_DIR / "object" / "vehicle_pedestrian_detection.pt"
SIGN_DETECTION_MODEL = MODELS_DIR / "traffic_sign" / "ts_det.pt"
SIGN_CLASSIFICATION_MODEL = MODELS_DIR / "traffic_sign" / "ts_class.h5"
LIGHT_DETECTION_CLASSIFICATION_MODEL = MODELS_DIR / "traffic_light" / "traffic_lights_yolov8x.pt"
SCNN_LANE_DETECTION_MODEL = MODELS_DIR / "scnn_lanedet" / "scnn.pth"

BEAMNG_HOME = r'C:\Users\user\Documents\beamng-tech\BeamNG.tech.v0.36.4.0'
