# RobotiArm-Classifier

An **educational robotic sorting system** featuring a 5-DOF PLA‑printed arm with MG996R/MG90S servos, controlled via Arduino, and powered by a Keras/TensorFlow CNN (MNIST, ≥97% accuracy) to recognize digits 0–9. The arm autonomously picks, sorts, and places 3D‑printed pieces.

---

<!-- HERO SECTION -->

<div align="center">
  <img src="https://github.com/user-attachments/assets/6a8d2995-9fcb-45e9-b205-adfd1ff9237b" alt="RobotiArm in action" width="45%" style="margin: 0 2%" />
  <img src="https://github.com/user-attachments/assets/f3bbdfea-95ce-4fa2-8be3-e27a78f6d5c5" alt="Vision & Control GUI" width="45%" style="margin: 0 2%" />
</div>

<div align="center" style="margin-top: 1rem;">
  <a href="https://youtu.be/gvfYf3450xA" target="_blank" style="text-decoration: none;">
    <img src="https://img.youtube.com/vi/gvfYf3450xA/maxresdefault.jpg" alt="Watch Demo" width="80%" />
    <h3>Watch the full demo ▶️</h3>
  </a>
</div>

---

## 📖 Overview

**RobotiArm Classifier** integrates:

1. **Mechanical & Electronics**

   * 5‑DOF PLA‑printed arm, six MG996R/MG90S servos, bearings.
   * 5 V / 15 A power supply for stable operation.

2. **Embedded Control (Arduino)**

   * `arduino_test.ino`: Arduino Uno/Nano firmware.
   * Serial protocol @9600 bps:
     `MOVE t1 t2 t3 t4 t5 t6`
     `GRIP OPEN | CLOSE`
   * Acknowledgments synchronize PC commands.

3. **Computer Vision & Deep Learning**

   * **CNN (Keras/TensorFlow)** trained on MNIST (>97%).
   * Preprocess: ROI crop, Otsu threshold, resize 28×28, normalize.
   * `number_classifier.py` handles inference.

4. **Python PC App**

   * **Manual**: `test_control.py` (Tkinter sliders, gripper control).
   * **Calibrate**: `test_cnn.py` (OpenCV trackbars, live prediction).
   * **Autonomous**: `main_robot.py` (capture → classify → pick‑place → home).

---

## 📂 Structure

```
RobotiArm-Classifier/
├── arduino_test.ino
├── cnn_mnist_final.keras
├── model/
│   └── cnn_mnist_best.keras
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

---

## ⚙️ Requirements

**Hardware:** Arduino Uno/Nano, 6×MG996R/MG90S servos, 5‑DOF arm, USB camera, 5 V≥15 A supply

**Software:** Python 3.8+, `tensorflow`, `keras`, `opencv-python`, `numpy`, `pygame`, `tkinter`, `pyserial`, Arduino IDE

---

## 🚀 Installation

```bash
git clone https://github.com/Eliasjapi-dev/RobotiArm-Classifier.git
cd RobotiArm-Classifier
pip install -r requirements.txt
```

Upload `arduino_test.ino` via Arduino IDE, then verify model files in root.

---

## ▶️ Usage

1. **Manual**: `python test_control.py --port COM3`
2. **Calibrate**: `python test_cnn.py --port COM3`
3. **Autonomous**: `python main_robot.py --port COM3`

---

## 🛠 Training

Open `Train_CNN_2.ipynb`, run all cells, save best model:

```python
model.save('model/cnn_mnist_best.keras')
```

Copy model & `cnn_mnist_class_indices.json` to root.

---

## 📄 License

MIT © \[Your Name]

---

## 🤝 Contribute

Fork → branch → PR → merge

---

## 📚 References

* Full report: `RobotiArm Classifier Documentation.docx/pdf`
* [Keras MNIST example](https://keras.io/examples/vision/mnist_convnet/)
* [Arduino docs](https://www.arduino.cc/)
* [TensorFlow](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
