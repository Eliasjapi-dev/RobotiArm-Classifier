# number_classifier.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os

def load_class_mapping(json_path):
    """Carga el mapeo de índices a clases desde un archivo JSON."""
    with open(json_path, 'r') as f:
        class_mapping = json.load(f)
    return {int(k): int(v) for k, v in class_mapping.items()}

def preprocess_image(thresh, bbox, img_width=28, img_height=28):
    """Extrae, invierte, redimensiona y normaliza la ROI individual para la predicción."""
    x, y, w, h = bbox
    margin = 5
    y_start = max(y - margin, 0)
    y_end = min(y + h + margin, thresh.shape[0])
    x_start = max(x - margin, 0)
    x_end = min(x + w + margin, thresh.shape[1])
    roi = thresh[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        return None
    roi = cv2.bitwise_not(roi)
    try:
        resized = cv2.resize(roi, (img_width, img_height))
    except Exception as e:
        print(f"Error resizing ROI: {e}")
        return None
    normalized = resized.astype('float32') / 255.0
    reshaped = np.expand_dims(normalized, axis=-1)
    reshaped = np.expand_dims(reshaped, axis=0)
    return reshaped

class NumberClassifier:
    def __init__(self, model_path, class_mapping_path):
        self.model = load_model(model_path)
        self.class_mapping = load_class_mapping(class_mapping_path)
        print("Modelo y mapeo de clases cargados exitosamente.")

    def predict_number(self, prepared_img):
        prediction = self.model.predict(prepared_img)
        class_idx = np.argmax(prediction, axis=1)[0]
        number = self.class_mapping.get(class_idx, None)
        return number

def get_number_from_frame(frame, classifier, roi_params, detection_params):
    GLOBAL_X, GLOBAL_Y, GLOBAL_W, GLOBAL_H = roi_params
    MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT, MIN_AREA, MAX_AREA = detection_params

    roi_global = frame[GLOBAL_Y:GLOBAL_Y + GLOBAL_H, GLOBAL_X:GLOBAL_X + GLOBAL_W]
    gray = cv2.cvtColor(roi_global, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = [cv2.boundingRect(cnt) for cnt in contours if (
        MIN_WIDTH < cv2.boundingRect(cnt)[2] < MAX_WIDTH and
        MIN_HEIGHT < cv2.boundingRect(cnt)[3] < MAX_HEIGHT and
        cv2.contourArea(cnt) > MIN_AREA and
        cv2.contourArea(cnt) < MAX_AREA)]
    rois = sorted(rois, key=lambda item: item[0])

    numbers = []
    for bbox in rois:
        prepared_img = preprocess_image(thresh, bbox)
        if prepared_img is not None:
            number = classifier.predict_number(prepared_img)
            if number is not None and 0 <= number <= 9:  # Aceptar solo números del 0 al 9
                numbers.append(number)
    return numbers
