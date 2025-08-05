# test_control.py

import serial
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading


class ServoController:
    def __init__(self, port='COM6', baudrate=9600, timeout=1):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)  # Esperar a que el Arduino reinicie
            print(f"Conectado a {port} a {baudrate} bps.")
            self.read_serial()  # Leer el mensaje de inicio
        except serial.SerialException as e:
            print(f"Error al conectar con el puerto serial: {e}")
            self.ser = None

    def send_command(self, command):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write((command + '\n').encode())
                print(f"Enviado: {command}")
                # Leer respuesta no bloqueante
                response = self.read_serial()
                return response
            except serial.SerialException as e:
                print(f"Error al enviar comando: {e}")
                return None
        else:
            print("Puerto serial no está abierto.")
            return None

    def read_serial(self):
        if self.ser and self.ser.in_waiting > 0:
            try:
                response = self.ser.readline().decode().strip()
                if response:
                    print(f"Arduino: {response}")
                    return response
            except serial.SerialException as e:
                print(f"Error al leer del serial: {e}")
        return None

    def move_servos(self, angles):
        """
        angles: lista o tupla de 6 ángulos (t1, t2, t3, t4, t5, t6)
        """
        if len(angles) != 6:
            print("Debe proporcionar exactamente 6 ángulos.")
            return None
        command = f"MOVE {' '.join(map(str, angles))}"
        return self.send_command(command)

    def grip_open(self):
        return self.send_command("GRIP OPEN")

    def grip_close(self):
        return self.send_command("GRIP CLOSE")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Puerto serial cerrado.")


class ServoGUI:
    def __init__(self, root, controller: ServoController):
        self.root = root
        self.controller = controller
        self.root.title("Control de Servos con Arduino")
        self.create_widgets()
        # self.current_angles = [90] * 6  # Ángulos iniciales
        self.current_angles = [180,165, 90, 90, 90, 180]
        self.update_log_thread = threading.Thread(target=self.update_log, daemon=True)
        self.update_log_thread.start()

    def create_widgets(self):
        # Crear un frame para los sliders
        sliders_frame = ttk.LabelFrame(self.root, text="Control de Servos")
        sliders_frame.pack(padx=10, pady=10, fill="x")

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
        move_button = ttk.Button(self.root, text="Mover Servos", command=self.move_servos_smooth)
        move_button.pack(pady=5)

        # Botones para abrir y cerrar el gripper
        gripper_frame = ttk.Frame(self.root)
        gripper_frame.pack(pady=5)

        open_button = ttk.Button(gripper_frame, text="Abrir Gripper", command=self.open_gripper)
        open_button.grid(row=0, column=0, padx=5)

        close_button = ttk.Button(gripper_frame, text="Cerrar Gripper", command=self.close_gripper)
        close_button.grid(row=0, column=1, padx=5)

        # Área de log para mostrar respuestas del Arduino
        log_frame = ttk.LabelFrame(self.root, text="Log de Comunicación")
        log_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=10, state='disabled')
        self.log_text.pack(padx=5, pady=5, fill="both", expand=True)

        # Botón para salir
        exit_button = ttk.Button(self.root, text="Salir", command=self.on_exit)
        exit_button.pack(pady=5)

    def update_label(self, idx):
        angle = self.slider_vars[idx].get()
        self.angle_labels[idx].config(text=str(int(angle)))

    def move_servos_smooth(self):
        target_angles = [int(var.get()) for var in self.slider_vars]
        print(f"Moviendo servos a: {target_angles}")
        threading.Thread(target=self.smooth_move, args=(self.current_angles, target_angles), daemon=True).start()

    def smooth_move(self, current, target, step=1, delay=0.02):
        """
        Realiza un movimiento suave desde los ángulos actuales hasta los ángulos objetivo.

        Args:
            current (list of int): Ángulos actuales.
            target (list of int): Ángulos objetivo.
            step (int): Incremento/decremento por paso.
            delay (float): Tiempo en segundos entre pasos.
        """
        max_steps = max(abs(t - c) for c, t in zip(current, target))
        steps = max_steps // step if step > 0 else 1

        if steps == 0:
            steps = 1

        new_angles = current.copy()

        for _ in range(steps):
            for i in range(6):
                if new_angles[i] < target[i]:
                    new_angles[i] = min(new_angles[i] + step, target[i])
                elif new_angles[i] > target[i]:
                    new_angles[i] = max(new_angles[i] - step, target[i])
            self.controller.move_servos(new_angles)
            self.current_angles = new_angles.copy()
            # Actualizar los sliders en la GUI
            for i, angle in enumerate(new_angles):
                self.slider_vars[i].set(angle)
            time.sleep(delay)

        # Asegurarse de que los ángulos finales sean exactamente los objetivos
        self.controller.move_servos(target)
        self.current_angles = target.copy()
        for i, angle in enumerate(target):
            self.slider_vars[i].set(angle)

    def open_gripper(self):
        response = self.controller.grip_open()
        if response:
            self.append_log(f"Arduino: {response}")
        else:
            self.append_log("No se recibió respuesta al abrir gripper.")

    def close_gripper(self):
        response = self.controller.grip_close()
        if response:
            self.append_log(f"Arduino: {response}")
        else:
            self.append_log("No se recibió respuesta al cerrar gripper.")

    def append_log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def update_log(self):
        while True:
            response = self.controller.read_serial()
            if response:
                self.append_log(f"Arduino: {response}")
            time.sleep(0.1)

    def on_exit(self):
        if messagebox.askokcancel("Salir", "¿Deseas salir de la aplicación?"):
            self.controller.close()
            self.root.destroy()


def main():
    # Reemplaza 'COM3' con el puerto correcto en tu sistema (e.g., '/dev/ttyACM0' en Linux)
    controller = ServoController(port='COM6', baudrate=9600, timeout=1)

    if not controller.ser:
        messagebox.showerror("Error", "No se pudo conectar con el Arduino. Verifica el puerto y la conexión.")
        return

    root = tk.Tk()
    gui = ServoGUI(root, controller)
    root.protocol("WM_DELETE_WINDOW", gui.on_exit)
    root.mainloop()


if __name__ == "__main__":
    main()
