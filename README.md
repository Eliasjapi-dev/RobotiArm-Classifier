# RobotiArm-Classifier
An educational robotic sorting system that uses a 5-DOF PLA-printed arm driven by MG996R/MG90S servos and controlled via Arduino. A Keras/TensorFlow CNN trained on MNIST (≥97% accuracy) recognizes digits 0–9, enabling the arm to autonomously pick, sort and place labeled 3D-printed pieces.

---

## 📖 Overview

**RobotiArm Classifier** brings together multiple disciplines:

1. **Mechanical & Electronics**

   * 5-DOF robotic arm printed in PLA, driven by six MG996R/MG90S servos and assembled with bearings.
   * 5 V / 15 A power supply to reliably drive all servos and the gripper.

2. **Embedded Control (Arduino)**

   * Firmware in `arduino_test.ino` running on an Arduino Uno/Nano.
   * Simple serial protocol at 9600 bps:

     ```
     MOVE t1 t2 t3 t4 t5 t6  
     GRIP OPEN | CLOSE  
     ```
   * Each command sends back a confirmation to synchronize with the PC application.

3. **Computer Vision & Deep Learning**

   * **CNN (Keras/TensorFlow)** trained on MNIST, achieving > 97 % accuracy.
   * Preprocessing: ROI cropping, Otsu thresholding, resizing to 28 × 28 px and normalization.
   * Inference wrapper in Python: `number_classifier.py`.

4. **PC Application in Python**

   * **Manual Control**: `test_control.py` — Tkinter GUI with sliders for servo angles and gripper control.
   * **Vision Calibration**: `test_cnn.py` — OpenCV trackbars to tune ROI and threshold parameters.
   * **Autonomous Operation**: `main_robot.py` —

     1. Captures video, detects and extracts the piece.
     2. Predicts the digit and maps it to a sorting bin.
     3. Moves the arm smoothly to pick, place, and release the piece.
     4. Returns to home position.

---

## 📂 Repository Structure

```
RobotiArm Classifier/
├── arduino_test/
│   └── arduino_test.ino
├── model/
│   └── cnn_mnist_best.keras
├── cnn_mnist_final.keras
├── cnn_mnist_class_indices.json
├── Train_CNN_2.ipynb
├── test_control.py
├── test_cnn.py
├── number_classifier.py
├── main_robot.py
├── RobotiArm Classifier Documentation.docx
├── RobotiArm Classifier Documentation.pdf
└── requirements.txt
```

* **`arduino_test.ino`**
  Arduino firmware: defines six Servo objects, parses serial commands, and sends acknowledgments.

* **`cnn_mnist_final.keras`** & **`model/cnn_mnist_best.keras`**
  Trained CNN model weights.

* **`cnn_mnist_class_indices.json`**
  Mapping from class indices to digit labels (0–9).

* **`Train_CNN_2.ipynb`**
  Jupyter notebook for training and evaluating the CNN on MNIST.

* **`number_classifier.py`**
  Loads the model and JSON mapping; implements `preprocess_image()` and `infer()` functions.

* **`test_control.py`**
  Tkinter-based GUI for manual servo and gripper control via serial.

* **`test_cnn.py`**
  Calibration tool using OpenCV trackbars for ROI and threshold adjustment with live prediction overlay.

* **`main_robot.py`**
  Main application integrating video capture, digit classification, and robotic arm control.

* **`RobotiArm Classifier Documentation.docx/pdf`**
  Comprehensive project report covering mechanical design, electronics, kinematics, test results, and future improvements.

---

## ⚙️ Requirements

* **Hardware**

  * Arduino Uno or Nano
  * 6 × MG996R/MG90S servos + gripper
  * 5-DOF PLA-printed robotic arm
  * USB camera (30 FPS)
  * 5 V ≥ 15 A power supply

* **Software**

  * Python 3.8+
  * Libraries: `tensorflow`, `keras`, `opencv-python`, `numpy`, `pygame`, `tkinter`, `pyserial`
  * Arduino IDE (to upload `arduino_test.ino`)

---

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Eliasjapi-dev/RobotiArm-Classifier.git
   cd RobotiaClassifier
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Upload firmware to Arduino**

   * Open `arduino_test.ino` in the Arduino IDE.
   * Select the correct board and port, then compile and upload.

4. **Verify model files**

   * Ensure `cnn_mnist_final.keras` and `cnn_mnist_class_indices.json` are present in the root directory.

---

## ▶️ Usage

### 1. Manual Control (Optional)

```bash
python test_control.py --port COM3
```

* Move servos with sliders, open/close gripper, and view logs in real time.

### 2. Vision Calibration

```bash
python test_cnn.py --port COM3
```

* Adjust the ROI and threshold settings, and observe live digit predictions.

### 3. Autonomous Sorting

```bash
python main_robot.py --port COM3
```

* Click **“Start Sorting Sequence”** in the GUI to begin the automatic pick-and-place routine.

---

## 🛠 Model Training

1. Open `Train_CNN_2.ipynb` in Jupyter Notebook.
2. Run all cells to preprocess data, build the CNN architecture, train, and validate.
3. Save the best-performing model:

   ```python
   model.save('model/cnn_mnist_best.keras')
   ```
4. Copy `cnn_mnist_best.keras` and `cnn_mnist_class_indices.json` back to the project root.

---

## 📄 License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository.  
2. Create a branch: `git checkout -b feature/new-feature`.  
3. Commit your changes: `git commit -m "Add new feature"`.  
4. Open a pull request.

---

## 📚 References

* Full project report: `RobotiArm Classifier Documentation.docx/pdf`
* Keras MNIST example: [https://keras.io/examples/vision/mnist\_convnet/](https://keras.io/examples/vision/mnist_convnet/)
* PySerial Arduino tutorial: [https://realpython.com/arduino-python-communication/](https://realpython.com/arduino-python-communication/)
* Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2005). Robot modeling and control. Wiley.
* Craig, J., & Prentice, P. (2005). Introduction to Robotics Mechanics and Control Third Edition. [https://www.changjiangcai.com/files/text-books/Introduction-to-Robotics-3rd-edition.pdf](https://www.changjiangcai.com/files/text-books/Introduction-to-Robotics-3rd-edition.pdf)
* Documentación de Arduino. [https://www.arduino.cc/](https://www.arduino.cc/)
* OpenCV Python. [https://opencv.org/](https://opencv.org/)
* TensorFlow/Keras. [https://www.tensorflow.org/](https://www.tensorflow.org/)
* Tkinter. [https://docs.python.org/3/library/tkinter.html](https://docs.python.org/3/library/tkinter.html)
