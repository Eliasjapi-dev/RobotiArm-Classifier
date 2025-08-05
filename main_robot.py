# main_robot.py

import serial
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from number_classifier import NumberClassifier, get_number_from_frame
import cv2
from PIL import Image, ImageTk
import numpy as np

# Asegúrate de que `test_control.py` y `number_classifier.py` estén en el mismo directorio o en el PYTHONPATH
from test_control import ServoController, ServoGUI

class VideoStream:
    """Clase para manejar la captura y actualización de video en la GUI."""
    def __init__(self, label_raw, label_processed, roi_params, detection_params, classifier):
        self.label_raw = label_raw
        self.label_processed = label_processed
        self.roi_params = roi_params
        self.detection_params = detection_params
        self.classifier = classifier
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam")
            messagebox.showerror("Error", "No se puede acceder a la webcam.")
        # Configuración de la cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Iniciar actualización de frames
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Mostrar frame bruto
            raw_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_pil = Image.fromarray(raw_image)
            raw_tk = ImageTk.PhotoImage(image=raw_pil)
            self.label_raw.config(image=raw_tk)
            self.label_raw.image = raw_tk

            # Procesar frame
            processed = self.process_frame(frame)
            if processed is not None:
                processed_pil = Image.fromarray(processed)
                processed_tk = ImageTk.PhotoImage(image=processed_pil)
                self.label_processed.config(image=processed_tk)
                self.label_processed.image = processed_tk
        self.label_raw.after(15, self.update_frame)  # Actualizar cada 15 ms (~66 fps)

    def process_frame(self, frame):
        GLOBAL_X, GLOBAL_Y, GLOBAL_W, GLOBAL_H = self.roi_params
        MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT, MIN_AREA, MAX_AREA = self.detection_params

        # Verificar límites de ROI
        frame_height, frame_width = frame.shape[:2]
        GLOBAL_X = min(GLOBAL_X, frame_width - 1)
        GLOBAL_Y = min(GLOBAL_Y, frame_height - 1)
        GLOBAL_W = min(GLOBAL_W, frame_width - GLOBAL_X)
        GLOBAL_H = min(GLOBAL_H, frame_height - GLOBAL_Y)

        # Definir la ROI Global
        roi_global = frame[GLOBAL_Y:GLOBAL_Y + GLOBAL_H, GLOBAL_X:GLOBAL_X + GLOBAL_W]

        # Dibujar ROI Global en el frame bruto
        cv2.rectangle(frame, (GLOBAL_X, GLOBAL_Y), (GLOBAL_X + GLOBAL_W, GLOBAL_Y + GLOBAL_H), (255, 0, 0), 2)

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
        numbers = []

        for bbox in rois:
            prepared_img = preprocess_image(thresh, bbox)
            if prepared_img is not None:
                number = self.classifier.predict_number(prepared_img)
                if number is not None and 0 <= number <= 9:
                    numbers.append(number)
                    # Dibujar rectángulo y número detectado
                    x, y, w, h = bbox
                    cv2.rectangle(roi_global, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(roi_global, str(number), (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return thresh

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

class RobotController:
    def __init__(self, servo_controller: ServoController, classifier: NumberClassifier, video_stream: VideoStream, predicted_number_var):
        self.servo = servo_controller
        self.classifier = classifier
        self.video_stream = video_stream
        self.cap = video_stream.cap  # Usar la misma captura de video
        self.predicted_number_var = predicted_number_var  # Variable para actualizar el número predicho en la GUI
        # Definir las posiciones predefinidas
        self.positions = {
            "initial": [180, 165, 90, 90, 90, 150],
            "object_safe": [150, 165, 100, 20, 90, 130],
            "object_pick": [150, 90, 130, 35, 90, 180],
            "object_pick_GC": [150, 90, 130, 35, 90, 130],
            "camera_safe": [180, 165, 100, 50, 90, 130],
            "camera_show": [180, 140, 100, 50, 90, 130],
            # Posiciones de clasificación Safe
            "classification_A_safe": [100, 165, 100, 50, 90, 130],
            "classification_B_safe": [70, 165, 100, 50, 90, 130],
            "classification_C_safe": [40, 165, 100, 50, 90, 130],
            "classification_D_safe": [10, 165, 100, 50, 90, 130],
            # Posiciones de clasificación Put
            "classification_A_put": [100, 120, 150, 50, 90, 130],
            "classification_B_put": [70, 120, 150, 50, 90, 130],
            "classification_C_put": [40, 120, 150, 50, 90, 130],
            "classification_D_put": [10, 120, 150, 50, 90, 130]
        }

        # Definir el mapeo de números a posiciones de clasificación
        self.number_to_classification = {
            0: "A",
            1: "A",
            2: "A",
            3: "B",
            4: "B",
            5: "B",
            6: "C",
            7: "C",
            8: "D",
            9: "D"
        }

        # Parámetros de ROI para la clasificación (ajusta según sea necesario)
        self.roi_params = (260, 150, 150, 150)  # (GLOBAL_X, GLOBAL_Y, GLOBAL_W, GLOBAL_H)
        self.detection_params = (5, 150, 50, 150, 200, 4000)  # (MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT, MIN_AREA, MAX_AREA)

        # Inicializar ángulos actuales
        self.current_angles = self.positions["initial"].copy()
        self.servo.move_servos(self.current_angles)  # Asegurar que el robot comience en la posición inicial

    def smooth_move(self, target_angles, step=1, delay=0.02):
        """
        Realiza un movimiento suave desde los ángulos actuales hasta los ángulos objetivo.

        Args:
            target_angles (list): Lista de 6 ángulos objetivo.
            step (int): Incremento/decremento por paso.
            delay (float): Tiempo en segundos entre pasos.
        """
        max_steps = max(abs(t - c) for c, t in zip(self.current_angles, target_angles))
        steps = max_steps // step if step > 0 else 1

        if steps == 0:
            steps = 1

        new_angles = self.current_angles.copy()

        for _ in range(steps):
            for i in range(6):
                if new_angles[i] < target_angles[i]:
                    new_angles[i] = min(new_angles[i] + step, target_angles[i])
                elif new_angles[i] > target_angles[i]:
                    new_angles[i] = max(new_angles[i] - step, target_angles[i])
            self.servo.move_servos(new_angles)
            self.current_angles = new_angles.copy()
            time.sleep(delay)

        # Asegurarse de que los ángulos finales sean exactamente los objetivos
        self.servo.move_servos(target_angles)
        self.current_angles = target_angles.copy()

    def execute_sequence(self):
        threading.Thread(target=self._sequence_thread, daemon=True).start()

    def _sequence_thread(self):
        try:
            # 1. Mover a Posición Inicial
            print("Moviendo a Posición Inicial")
            self.smooth_move(self.positions["initial"])
            time.sleep(0.5)  # Esperar a que el movimiento se complete

            # 2. Mover a Posición Safe del Objeto
            print("Moviendo a Posición Safe del Objeto")
            self.smooth_move(self.positions["object_safe"])
            time.sleep(0.5)

            # 3. Mover a Posición Pick para recoger el objeto
            print("Moviendo a Posición Pick")
            self.smooth_move(self.positions["object_pick"])
            time.sleep(0.5)

            print("Moviendo a Posición Pick GC")
            self.smooth_move(self.positions["object_pick_GC"])
            time.sleep(0.5)

            # # 4. Cerrar el Gripper
            # print("Cerrando Gripper")
            # self.servo.grip_close()
            # time.sleep(1)  # Esperar a que el gripper cierre

            # 5. Mover a Posición Safe del Objeto
            print("Moviendo de vuelta a Posición Safe del Objeto")
            self.smooth_move(self.positions["object_safe"])
            time.sleep(0.5)

            # 6. Mover a Posición de la Cámara Safe
            print("Moviendo a Posición de la Cámara Safe")
            self.smooth_move(self.positions["camera_safe"])
            time.sleep(0.5)

            # 7. Mover a Posición de la Cámara Show
            print("Moviendo a Posición de la Cámara Show")
            self.smooth_move(self.positions["camera_show"])
            time.sleep(1)  # Esperar a que el servo se estabilice

            # 8. Esperar para ajustar la cámara antes de capturar
            print("Ajusta la cámara. La detección se realizará en 5 segundos.")
            time.sleep(5)

            # 9. Capturar y Clasificar el Número
            print("Capturando imagen para clasificación")
            ret, frame = self.cap.read()
            if not ret:
                print("Error: No se pudo capturar la imagen.")
                messagebox.showerror("Error", "No se pudo capturar la imagen.")
                return

            number = get_number_from_frame(frame, self.classifier, self.roi_params, self.detection_params)
            if not number:
                print("No se detectó un número válido.")
                messagebox.showerror("Error", "No se detectó un número válido.")
                self.predicted_number_var.set("No se detectó un número válido.")
                return

            # Asumimos que solo se detecta un número por pieza
            detected_number = number[0]
            print(f"Número detectado: {detected_number}")
            self.predicted_number_var.set(f"Número detectado: {detected_number}")

            # Obtener la clasificación correspondiente
            classification_label = self.number_to_classification.get(detected_number, None)
            if classification_label is None:
                print(f"Número {detected_number} no tiene asignada una ubicación de clasificación.")
                messagebox.showerror("Error", f"Número {detected_number} no tiene asignada una ubicación de clasificación.")
                return

            # Determinar las claves de posición para la clasificación
            classification_safe_key = f"classification_{classification_label}_safe"
            classification_put_key = f"classification_{classification_label}_put"

            if classification_safe_key in self.positions and classification_put_key in self.positions:
                print(f"Moviendo a Posición Safe para Clasificación {classification_label}")
                self.smooth_move(self.positions[classification_safe_key])
                time.sleep(0.5)

                print(f"Moviendo a Posición Put para Clasificación {classification_label}")
                self.smooth_move(self.positions[classification_put_key])
                time.sleep(0.5)

                # Abrir el Gripper para soltar la pieza
                print("Abriendo Gripper")
                self.servo.grip_open()
                time.sleep(1)  # Esperar a que el gripper abra

                # Volver a la Posición Safe de Clasificación
                print(f"Volviendo a Posición Safe para Clasificación {classification_label}")
                self.smooth_move(self.positions[classification_safe_key])
                time.sleep(0.5)
            else:
                print(f"No hay posición definida para la clasificación {classification_label}.")
                messagebox.showerror("Error", f"No hay posición definida para la clasificación {classification_label}.")
                return

            # 10. Volver a Posición Inicial
            print("Volviendo a Posición Inicial")
            self.smooth_move(self.positions["initial"])
            time.sleep(0.5)

            messagebox.showinfo("Éxito", f"Pieza clasificada y colocada en la posición {classification_label}.")

        except Exception as e:
            print(f"Error durante la secuencia: {e}")
            messagebox.showerror("Error", f"Error durante la secuencia: {e}")

class CustomServoGUI(tk.Frame):
    """Clase GUI personalizada para incluir video feed y controles reorganizados."""
    def __init__(self, root, controller: ServoController, classifier: NumberClassifier, roi_params, detection_params):
        super().__init__(root)
        self.root = root
        self.servo_controller = controller
        self.classifier = classifier
        self.roi_params = roi_params
        self.detection_params = detection_params

        # Variable para mostrar el número predicho
        self.predicted_number_var = tk.StringVar()
        self.predicted_number_var.set("Número predicho: N/A")

        # Crear la estructura de la GUI
        self.create_widgets()

        # Crear el objeto VideoStream
        self.video_stream = VideoStream(self.raw_video_label, self.processed_video_label, roi_params, detection_params, classifier)

        # Asignar el objeto VideoStream a RobotController
        self.robot_controller = RobotController(controller, classifier, self.video_stream, self.predicted_number_var)

    def create_widgets(self):
        # Frame principal
        self.pack(fill="both", expand=True)

        # Frame para controles del robot
        control_frame = ttk.LabelFrame(self, text="Control del Robot")
        control_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Botón para iniciar la secuencia
        start_button = ttk.Button(control_frame, text="Iniciar Secuencia de Clasificación", command=self.start_sequence)
        start_button.pack(pady=10)

        # Mostrar el número predicho
        predicted_number_label = ttk.Label(control_frame, textvariable=self.predicted_number_var, font=("Helvetica", 14))
        predicted_number_label.pack(pady=10)

        # Frame para controles de servos
        sliders_frame = ttk.LabelFrame(control_frame, text="Control de Servos")
        sliders_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.sliders = []
        self.slider_vars = []
        self.angle_labels = []

        for i in range(6):
            var = tk.IntVar(value=90)
            self.slider_vars.append(var)
            slider = ttk.Scale(
                sliders_frame,
                from_=0,
                to=180,
                orient='horizontal',
                variable=var,
                command=lambda val, idx=i: self.update_label(idx)
            )
            slider.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            sliders_frame.columnconfigure(1, weight=1)

            label = ttk.Label(sliders_frame, text=f"Servo {i + 1}:")
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")

            angle_label = ttk.Label(sliders_frame, text="90")
            angle_label.grid(row=i, column=2, padx=5, pady=5, sticky="w")
            self.sliders.append(slider)
            self.angle_labels.append(angle_label)

        # Botón para mover los servos
        move_button = ttk.Button(sliders_frame, text="Mover Servos", command=self.move_servos_smooth)
        move_button.grid(row=6, column=0, columnspan=3, pady=10)

        # Botones para abrir y cerrar el gripper
        gripper_frame = ttk.Frame(sliders_frame)
        gripper_frame.grid(row=7, column=0, columnspan=3, pady=10)

        open_button = ttk.Button(gripper_frame, text="Abrir Gripper", command=self.open_gripper)
        open_button.grid(row=0, column=0, padx=5)

        close_button = ttk.Button(gripper_frame, text="Cerrar Gripper", command=self.close_gripper)
        close_button.grid(row=0, column=1, padx=5)

        # Frame para el video
        video_frame = ttk.LabelFrame(self, text="Video")
        video_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Label para el feed de la webcam
        self.raw_video_label = ttk.Label(video_frame)
        self.raw_video_label.pack(side="left", padx=5, pady=5, expand=True)

        # Label para la imagen procesada
        self.processed_video_label = ttk.Label(video_frame)
        self.processed_video_label.pack(side="right", padx=5, pady=5, expand=True)

    def update_label(self, idx):
        angle = self.slider_vars[idx].get()
        self.angle_labels[idx].config(text=str(int(angle)))

    def move_servos_smooth(self):
        target_angles = [int(var.get()) for var in self.slider_vars]
        print(f"Moviendo servos a: {target_angles}")
        threading.Thread(target=self.smooth_move, args=(target_angles,), daemon=True).start()

    def smooth_move(self, target_angles, step=1, delay=0.02):
        """
        Realiza un movimiento suave desde los ángulos actuales hasta los ángulos objetivo.

        Args:
            target_angles (list of int): Ángulos objetivo.
            step (int): Incremento/decremento por paso.
            delay (float): Tiempo en segundos entre pasos.
        """
        max_steps = max(abs(t - c) for c, t in zip(self.robot_controller.current_angles, target_angles))
        steps = max_steps // step if step > 0 else 1

        if steps == 0:
            steps = 1

        new_angles = self.robot_controller.current_angles.copy()

        for _ in range(steps):
            for i in range(6):
                if new_angles[i] < target_angles[i]:
                    new_angles[i] = min(new_angles[i] + step, target_angles[i])
                elif new_angles[i] > target_angles[i]:
                    new_angles[i] = max(new_angles[i] - step, target_angles[i])
            self.servo_controller.move_servos(new_angles)
            self.robot_controller.current_angles = new_angles.copy()
            time.sleep(delay)

        # Asegurarse de que los ángulos finales sean exactamente los objetivos
        self.servo_controller.move_servos(target_angles)
        self.robot_controller.current_angles = target_angles.copy()

    def open_gripper(self):
        self.servo_controller.grip_open()

    def close_gripper(self):
        self.servo_controller.grip_close()

    def start_sequence(self):
        """Iniciar la secuencia de clasificación."""
        self.robot_controller.execute_sequence()

    def on_exit(self):
        """Manejar la salida de la aplicación."""
        if messagebox.askokcancel("Salir", "¿Deseas salir de la aplicación?"):
            if self.video_stream.cap.isOpened():
                self.video_stream.cap.release()
            self.servo_controller.close()
            self.root.destroy()

def main():
    # Inicializar el controlador de servos
    servo_controller = ServoController(port='COM6', baudrate=9600, timeout=1)
    if not servo_controller.ser:
        messagebox.showerror("Error", "No se pudo conectar con el Arduino. Verifica el puerto y la conexión.")
        return

    # Inicializar el clasificador
    classifier = NumberClassifier(model_path='model/cnn_mnist_best.keras',
                                  class_mapping_path='cnn_mnist_class_indices.json')

    # Inicializar la GUI
    root = tk.Tk()
    root.title("Control de Robot Clasificador")

    # Definir los parámetros de ROI y detección
    roi_params = (260, 150, 150, 150)  # (GLOBAL_X, GLOBAL_Y, GLOBAL_W, GLOBAL_H)
    detection_params = (5, 150, 50, 150, 200, 4000)  # (MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT, MIN_AREA, MAX_AREA)

    # Crear la interfaz de control de servos y video
    gui = CustomServoGUI(root, servo_controller, classifier, roi_params, detection_params)

    root.protocol("WM_DELETE_WINDOW", gui.on_exit)
    root.mainloop()

if __name__ == "__main__":
    main()
