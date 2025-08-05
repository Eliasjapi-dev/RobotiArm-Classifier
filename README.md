# RobotiArm-Classifier

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg) ![Python Version](https://img.shields.io/badge/Python-3.8%2B-green.svg)

An **educational robotic sorting system** featuring a 5‑DOF PLA‑printed arm with MG996R/MG90S servos, controlled via Arduino, and powered by a Keras/TensorFlow CNN (MNIST, ≥97% accuracy) to recognize digits 0–9. The arm autonomously picks, sorts, and places 3D‑printed pieces.

---

<!-- HERO SECTION -->

<div align="center">
  <img src="https://github.com/user-attachments/assets/6a8d2995-9fcb-45e9-b205-adfd1ff9237b" alt="Action Shot" width="350px" style="margin: 0 10px; border-radius: 8px;" />
  <img src="https://github.com/user-attachments/assets/f3bbdfea-95ce-4fa2-8be3-e27a78f6d5c5" alt="GUI Preview" width="350px" style="margin: 0 10px; border-radius: 8px;" />
</div>

<div align="center" style="margin-top: 20px;">
  <a href="https://youtu.be/gvfYf3450xA" target="_blank">
    <img src="https://img.youtube.com/vi/gvfYf3450xA/maxresdefault.jpg" alt="Demo Video Thumbnail" width="700px" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);" />
  </a>
  <p><a href="https://youtu.be/gvfYf3450xA" target="_blank" style="font-size: 1.2rem; font-weight: bold; text-decoration: none;">▶️ Watch the Full Demo</a></p>
</div>

---

## 📖 Overview

**RobotiArm Classifier** integrates:

| Component                    | Description                                                                                        |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| **Mechanical & Electronics** | 5‑DOF PLA‑printed arm + MG996R/MG90S servos, powered by 5 V/15 A supply                            |
| **Embedded Control**         | Arduino Uno/Nano firmware (`arduino_test.ino`) with simple serial commands                         |
| **Computer Vision**          | ROI cropping, Otsu threshold, CNN inference (Keras/TensorFlow, ≥97% MNIST)                         |
| **PC Application**           | Python GUIs: manual (`test_control.py`), calibration (`test_cnn.py`), autonomous (`main_robot.py`) |

---

## 📂 Repository Structure

```text
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

---

## ⚙️ Requirements

<table>
  <tr>
    <th align="left">Hardware</th>
    <th align="left">Software</th>
  </tr>
  <tr>
    <td>
      • Arduino Uno/Nano<br>
      • 6× MG996R/MG90S servos + gripper<br>
      • 5‑DOF PLA arm<br>
      • USB camera (30 FPS)<br>
      • 5 V / 15 A power supply
    </td>
    <td>
      • Python 3.8+<br>
      • `tensorflow`<br>
      • `keras`<br>
      • `opencv-python`<br>
      • `numpy`<br>
      • `pygame`<br>
      • `tkinter`<br>
      • `pyserial`<br>
      • Arduino IDE
    </td>
  </tr>
</table>

---

## 🚀 Installation

```bash
git clone https://github.com/Eliasjapi-dev/RobotiArm-Classifier.git
cd RobotiArm-Classifier
pip install -r requirements.txt
```

1. Upload `arduino_test.ino` to Arduino using the IDE
2. Verify `cnn_mnist_final.keras` & `cnn_mnist_class_indices.json` are in the root

---

## ▶️ Usage

| Mode        | Command                              |
| ----------- | ------------------------------------ |
| Manual      | `python test_control.py --port COM3` |
| Calibration | `python test_cnn.py --port COM3`     |
| Autonomous  | `python main_robot.py --port COM3`   |

---

## 🛠 Model Training

1. Open `Train_CNN_2.ipynb` in Jupyter
2. Run all cells (data prep, model train, eval)
3. Save best model:

   ```python
   model.save('model/cnn_mnist_best.keras')
   ```
4. Copy model & `cnn_mnist_class_indices.json` to root

---

## 📄 License & Contribution

Licensed under the **MIT License**. See [LICENSE](LICENSE).

Contributions welcome: fork, branch, PR, merge!

---

## 📚 References

* Full project report: `RobotiArm Classifier Documentation.docx/pdf`
* Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2005). *Robot Modeling and Control*. Wiley.
* Craig, J., & Prentice, P. (2005). *Introduction to Robotics Mechanics and Control* (3rd ed.). [PDF](https://www.changjiangcai.com/files/text-books/Introduction-to-Robotics-3rd-edition.pdf)
* Arduino Documentation: [https://www.arduino.cc/](https://www.arduino.cc/)
* OpenCV Python: [https://opencv.org/](https://opencv.org/)
* TensorFlow/Keras: [https://www.tensorflow.org/](https://www.tensorflow.org/)
* Tkinter: [https://docs.python.org/3/library/tkinter.html](https://docs.python.org/3/library/tkinter.html)
