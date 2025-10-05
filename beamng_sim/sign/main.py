from beamng_sim.sign.detect_classify import combined_sign_detection_classification

DEFAULT_CONFIDENCE_THRESHOLD = 0.45

def process_frame(img, draw_detections=True, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    try:
        detections = combined_sign_detection_classification(img)
        
        if not detections:
            detections = []
        
        filtered_detections = [det for det in detections if det['detection_confidence'] >= confidence_threshold]
            
        result_img = img
        
        if draw_detections:
            import cv2
            result_img = img.copy()
            for det in filtered_detections:
                x1,y1,x2,y2 = det['bbox']
                
                classification = det.get('classification', 'Unknown')
                confidence = det.get('classification_confidence', 0.0)
                
                label = f"{classification} ({confidence:.2f})"
                cv2.rectangle(result_img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(result_img, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        return filtered_detections, result_img
    except Exception as e:
        print(f"Error processing sign frame: {e}")
        return [], img
