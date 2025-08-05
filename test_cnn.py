# test_cnn_with_sliders.py

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
    margin = 10
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
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        if not os.path.exists(class_mapping_path):
            raise FileNotFoundError(f"Mapa de clases no encontrado en {class_mapping_path}")

        self.model = load_model(model_path)
        self.class_mapping = load_class_mapping(class_mapping_path)
        print("Modelo y mapeo de clases cargados exitosamente.")

    def predict_number(self, prepared_img):
        prediction = self.model.predict(prepared_img)
        class_idx = np.argmax(prediction, axis=1)[0]
        number = self.class_mapping.get(class_idx, None)
        confidence = np.max(prediction, axis=1)[0]
        return number, confidence


def nothing(x):
    pass


def create_trackbars_roi():
    """Crea las trackbars para ajustar los parámetros de la ROI Global."""
    cv2.namedWindow("Ajustes ROI Global")

    # Parámetros de la ROI Global
    cv2.createTrackbar("GLOBAL_X", "Ajustes ROI Global", 260, 640, nothing)
    cv2.createTrackbar("GLOBAL_Y", "Ajustes ROI Global", 150, 480, nothing)
    cv2.createTrackbar("GLOBAL_W", "Ajustes ROI Global", 150, 640, nothing)
    cv2.createTrackbar("GLOBAL_H", "Ajustes ROI Global", 150, 480, nothing)


def create_trackbars_detection():
    """Crea las trackbars para ajustar los parámetros de Detección."""
    cv2.namedWindow("Ajustes Detección")

    # Parámetros de Detección
    cv2.createTrackbar("MIN_WIDTH", "Ajustes Detección", 5, 200, nothing)
    cv2.createTrackbar("MAX_WIDTH", "Ajustes Detección", 150, 500, nothing)
    cv2.createTrackbar("MIN_HEIGHT", "Ajustes Detección", 50, 200, nothing)
    cv2.createTrackbar("MAX_HEIGHT", "Ajustes Detección", 150, 500, nothing)
    cv2.createTrackbar("MIN_AREA", "Ajustes Detección", 200, 10000, nothing)
    cv2.createTrackbar("MAX_AREA", "Ajustes Detección", 4000, 20000, nothing)


def get_trackbar_values():
    """Obtiene los valores actuales de las trackbars."""
    # ROI Global
    GLOBAL_X = cv2.getTrackbarPos("GLOBAL_X", "Ajustes ROI Global")
    GLOBAL_Y = cv2.getTrackbarPos("GLOBAL_Y", "Ajustes ROI Global")
    GLOBAL_W = cv2.getTrackbarPos("GLOBAL_W", "Ajustes ROI Global")
    GLOBAL_H = cv2.getTrackbarPos("GLOBAL_H", "Ajustes ROI Global")

    # Parámetros de Detección
    MIN_WIDTH = cv2.getTrackbarPos("MIN_WIDTH", "Ajustes Detección")
    MAX_WIDTH = cv2.getTrackbarPos("MAX_WIDTH", "Ajustes Detección")
    MIN_HEIGHT = cv2.getTrackbarPos("MIN_HEIGHT", "Ajustes Detección")
    MAX_HEIGHT = cv2.getTrackbarPos("MAX_HEIGHT", "Ajustes Detección")
    MIN_AREA = cv2.getTrackbarPos("MIN_AREA", "Ajustes Detección")
    MAX_AREA = cv2.getTrackbarPos("MAX_AREA", "Ajustes Detección")

    # Validaciones para evitar errores
    if MAX_WIDTH < MIN_WIDTH:
        MAX_WIDTH = MIN_WIDTH + 1
    if MAX_HEIGHT < MIN_HEIGHT:
        MAX_HEIGHT = MIN_HEIGHT + 1
    if MAX_AREA < MIN_AREA:
        MAX_AREA = MIN_AREA + 1

    roi_params = (GLOBAL_X, GLOBAL_Y, GLOBAL_W, GLOBAL_H)
    detection_params = (MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT, MIN_AREA, MAX_AREA)

    return roi_params, detection_params


def main():
    # Rutas al modelo y al mapeo de clases
    MODEL_PATH = 'model/cnn_mnist_best.keras'  # Actualiza esta ruta si es necesario
    CLASS_MAPPING_PATH = 'cnn_mnist_class_indices.json'  # Actualiza esta ruta si es necesario

    # Inicializar el clasificador
    try:
        classifier = NumberClassifier(model_path=MODEL_PATH, class_mapping_path=CLASS_MAPPING_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Iniciar captura de video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede abrir la webcam.")
        return

    # Configuración de la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Crear trackbars para ajustar parámetros
    create_trackbars_roi()
    create_trackbars_detection()

    print("Iniciando la captura de video. Presiona 'q' para salir.")
    print(
        "Ajusta los sliders en las ventanas 'Ajustes ROI Global' y 'Ajustes Detección' para modificar los parámetros.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la webcam.")
            break

        # Obtener valores actuales de los sliders
        roi_params, detection_params = get_trackbar_values()
        GLOBAL_X, GLOBAL_Y, GLOBAL_W, GLOBAL_H = roi_params
        MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT, MIN_AREA, MAX_AREA = detection_params

        # Definir la ROI Global
        frame_height, frame_width = frame.shape[:2]
        GLOBAL_X = min(GLOBAL_X, frame_width - 1)
        GLOBAL_Y = min(GLOBAL_Y, frame_height - 1)
        GLOBAL_W = min(GLOBAL_W, frame_width - GLOBAL_X)
        GLOBAL_H = min(GLOBAL_H, frame_height - GLOBAL_Y)

        roi_global = frame[GLOBAL_Y:GLOBAL_Y + GLOBAL_H, GLOBAL_X:GLOBAL_X + GLOBAL_W]

        # Dibujar ROI Global en el frame bruto
        annotated_frame = frame.copy()
        cv2.rectangle(annotated_frame, (GLOBAL_X, GLOBAL_Y), (GLOBAL_X + GLOBAL_W, GLOBAL_Y + GLOBAL_H), (255, 0, 0), 2)

        # Preprocesar la ROI Global
        gray = cv2.cvtColor(roi_global, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Encontrar contornos
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = [cv2.boundingRect(cnt) for cnt in contours if (
                MIN_WIDTH < cv2.boundingRect(cnt)[2] < MAX_WIDTH and
                MIN_HEIGHT < cv2.boundingRect(cnt)[3] < MAX_HEIGHT and
                cv2.contourArea(cnt) > MIN_AREA and
                cv2.contourArea(cnt) < MAX_AREA)]
        rois = sorted(rois, key=lambda item: item[0])

        # Lista para almacenar números detectados
        detected_numbers = []

        for bbox in rois:
            prepared_img = preprocess_image(thresh, bbox)
            if prepared_img is not None:
                number, confidence = classifier.predict_number(prepared_img)
                if number is not None and 0 <= number <= 9:
                    detected_numbers.append((number, confidence))
                    # Dibujar rectángulo y número detectado en la ROI Global
                    x, y, w, h = bbox
                    cv2.rectangle(roi_global, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(roi_global, f"{number} ({confidence * 100:.1f}%)",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2)

        # Mostrar el número predicho en la pantalla principal
        if detected_numbers:
            # Si hay múltiples números, puedes decidir cómo manejarlos. Aquí mostramos el primero.
            predicted_number, pred_confidence = detected_numbers[0]
            cv2.putText(annotated_frame, f"Prediccion: {predicted_number} ({pred_confidence * 100:.1f}%)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)
        else:
            cv2.putText(annotated_frame, "Prediccion: N/A",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2)

        # Mostrar la ROI Global procesada
        cv2.imshow("ROI Global", roi_global)

        # Mostrar el frame anotado con predicciones
        cv2.imshow("Clasificacion de Numeros", annotated_frame)

        # Opcional: Mostrar la imagen umbralizada para debug
        # cv2.imshow("Umbral", thresh)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
