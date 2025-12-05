import cv2 as cv
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import sys
from PIL import Image
from beamng_sim.vehicle_obstacle.vehicle_obstacle_detection import detect_vehicles_pedestrians
from config.config import SIGN_DETECTION_MODEL, SIGN_CLASSIFICATION_MODEL

from tensorflow.keras.models import load_model

IMG_SIZE = (48, 48)
SIGN_MODEL_PATH = str(SIGN_DETECTION_MODEL)
SIGN_CLASSIFY_MODEL_PATH = str(SIGN_CLASSIFICATION_MODEL)

# GTSRB class names (hardcoded)
SIGN_CLASSES = { 
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow',
    31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons' 
}

def get_models_dict():
    try:
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'MODELS'):
            return main_module.MODELS
        return None
    except:
        return None

def preprocess_img(img):
    """
    Preprocessing function matching the training script.
    Args:
        img: numpy array image (RGB)
    Returns:
        Preprocessed image (48x48, normalized)
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty")
        
    # Convert to HSV
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    # Histogram equalization on V channel
    hsv[:,:,2] = cv.equalizeHist(hsv[:,:,2])
    # Convert back to RGB
    img = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    # Resize to 48x48
    img = cv.resize(img, IMG_SIZE)
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    return img

# Build class descriptions from hardcoded SIGN_CLASSES
class_descriptions = ["Unknown Class"] * 43
for class_id, description in SIGN_CLASSES.items():
    if 0 <= class_id < 43:
        class_descriptions[class_id] = description

def classify_sign_crop(sign_crop):
    try:
        # Preprocess the crop
        img = preprocess_img(sign_crop)
        img = np.expand_dims(img, axis=0)

        models_dict = get_models_dict()
        if models_dict is not None and 'sign_classify' in models_dict:
            classification_model = models_dict['sign_classify']
        else:
            classification_model = load_model(SIGN_CLASSIFY_MODEL_PATH)

        pred = classification_model.predict(img, verbose=0)
        class_idx = np.argmax(pred[0])
        class_confidence = float(pred[0][class_idx])

        if 0 <= class_idx < len(class_descriptions):
            classification = class_descriptions[class_idx]
        else:
            classification = f"Class {class_idx}"

        return {
            'class': classification,
            'confidence': class_confidence,
            'class_index': int(class_idx)
        }
        
    except Exception as e:
        print(f"Error in classify_sign_crop: {e}")
        import traceback
        traceback.print_exc()
        return {
            'class': 'Classification Error',
            'confidence': 0.0,
            'class_index': -1
        }

def img_preprocessing(frame):
    img = frame.copy()
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_classify_sign(frame):
    models_dict = get_models_dict()
    
    if models_dict is not None and 'sign_detect' in models_dict:
        detection_model = models_dict['sign_detect']
    else:
        detection_model = YOLO(SIGN_MODEL_PATH)
        print(f"Warning: Loading sign detection model from scratch - slower!")
    
    if models_dict is not None and 'sign_classify' in models_dict:
        classification_model = models_dict['sign_classify']
    else:
        classification_model = tf.keras.models.load_model(SIGN_CLASSIFY_MODEL_PATH)
        print(f"Warning: Loading sign classification model from scratch - slower!")

    results = detection_model(frame, conf=0.2)

    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            class_id = int(box.cls[0])
            class_name = detection_model.names[class_id]
            confidence = float(box.conf[0])
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                sign_crop = frame[y1:y2, x1:x2]
                
                try:
                    classification_result = classify_sign_crop(sign_crop)
                    classification = classification_result['class']
                    class_confidence = classification_result['confidence']
                    class_idx = classification_result['class_index']
                    
                    if 0 <= class_idx < len(class_descriptions):
                        classification = class_descriptions[class_idx]
                    else:
                        classification = f"Class {class_idx}"
                        
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'detection_class': class_name,
                        'detection_confidence': confidence,
                        'classification': classification,
                        'classification_confidence': class_confidence
                    })
                    
                except Exception as e:
                    print(f"Error during classification: {e}")
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'detection_class': class_name,
                        'detection_confidence': confidence,
                        'classification': "Classification failed",
                        'classification_confidence': 0.0
                    })
            
    return detections

def combined_sign_detection_classification(frame):
    sign_model_detections = detect_classify_sign(frame)
    print(f"Sign model detected {len(sign_model_detections)} signs")

    vehicle_model_detections = detect_vehicles_pedestrians(frame, include_traffic_lights=False, include_traffic_signs=True)

    classes_found = set()
    for detection in vehicle_model_detections:
        classes_found.add(detection['class'])

    vehicle_model_sign_detections = []
    for detection in vehicle_model_detections:
        if ('traffic sign' in detection['class'].lower() or 
            'traffic signs' in detection['class'].lower() or 
            'sign' in detection['class'].lower()):
            detection['source'] = 'vehicle_model'
            detection['class'] = 'unknown'
            vehicle_model_sign_detections.append(detection)
    
    print(f"Vehicle model detected {len(vehicle_model_sign_detections)} signs")

    final_detections = []

    SIGN_MODEL_THRESHOLD = 0.4  # Internal threshold for sign model detections
    VEHICLE_MODEL_THRESHOLD = 0.4  # Internal threshold for vehicle model detections
    CLASS_CONFIDENCE_THRESHOLD = 0.4  # Internal threshold for classification confidence

    for sign_det in sign_model_detections:
        if sign_det['detection_confidence'] > SIGN_MODEL_THRESHOLD:
            sign_det['source'] = 'sign_model'
            sign_det['verified'] = False
            final_detections.append(sign_det)

    for veh_det in vehicle_model_sign_detections:
        if veh_det['confidence'] > VEHICLE_MODEL_THRESHOLD:
            try:
                x1, y1, x2, y2 = veh_det['bbox']
                sign_crop = frame[y1:y2, x1:x2]
                
                classification_result = classify_sign_crop(sign_crop)
                classification = classification_result['class']
                class_confidence = classification_result['confidence']
                class_idx = classification_result['class_index']
                
                if class_confidence > CLASS_CONFIDENCE_THRESHOLD:
                    enhanced_det = {
                        'bbox': veh_det['bbox'],
                        'detection_class': veh_det['class'],
                        'detection_confidence': veh_det['confidence'],
                        'classification': classification,
                        'classification_confidence': class_confidence,
                        'source': 'vehicle_model',
                        'verified': False,
                        'class_idx': class_idx
                    }
                    
                    final_detections.append(enhanced_det)
                
            except Exception as e:
                print(f"Error classifying vehicle model sign: {e}")
    
    filtered_detections = []
    for i, det1 in enumerate(final_detections):
        x1_1, y1_1, x2_1, y2_1 = det1['bbox']
        center1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
        
        if 'skip' in det1 and det1['skip']:
            continue
        
        filtered_detections.append(det1)
        
        for j in range(i+1, len(final_detections)):
            det2 = final_detections[j]
            x1_2, y1_2, x2_2, y2_2 = det2['bbox']
            center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
            
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            if distance < 50:
                if det1['detection_confidence'] < det2['detection_confidence']:
                    filtered_detections.pop()
                    filtered_detections.append(det2)
                    det1['skip'] = True
                    break
                else:
                    det2['skip'] = True
    
    for det in filtered_detections:
        if 'classification' not in det or det['classification'] in ['unknown', 'traffic sign', 'traffic signs']:
            if 'class_idx' in det and 0 <= det['class_idx'] < len(class_descriptions):
                det['classification'] = class_descriptions[det['class_idx']]
    
    print(f"Combined sign detection returning {len(filtered_detections)} signs")
    return filtered_detections
