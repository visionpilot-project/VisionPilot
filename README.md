# 🚗 Self-Driving Car Simulation: Computer Vision, Deep Learning & Real-Time Perception

![GitHub stars](https://img.shields.io/github/stars/Julian1777/self-driving-project?style=social)

A modular Python project for autonomous driving research and prototyping. This system combines traditional computer vision and state-of-the-art deep learning (CNN, U-Net, YOLO, SCNN) to tackle:

- 🛣️ Lane detection (Hough Transform, SCNN, city/highway scenarios)
- 🛑 Traffic sign classification & detection (CNN, YOLOv8, GTRSB, LISA, Mapillary)
- 🚦 Traffic light detection & classification (YOLOv8, DLDT, LISA)
- 🚗 Vehicle & pedestrian detection and recognition (YOLOv8, SCNN, BDD100K)
- 🧠 Multi-model inference, real-time simulation, and visualization (Pygame, CARLA)

Features robust training pipelines, multi-model inference, and a flexible folder structure for easy experimentation and extension.

## 🎥 Demo

| Lane Detection | Sign Recognition | Traffic Light Detection |
|----------------|------------------|--------------------------|
| ![lane](assets/lane.gif) | ![sign](assets/sign.gif) | ![light](assets/light.gif) |


## 🔧 Features

-  Lane detection with SCNN and OpenCV (comparison)
-  Traffic sign classification using CNN
-  Traffic light detection (YOLO) + classification
-  Video-based inference pipeline
-  Multi-window simulation using Pygame
- 🚀 Coming soon: CARLA integration & real-time testing

## 🛠️ Built With

- TensorFlow / Keras
- OpenCV
- YOLOv8 (Ultralytics)
- Python
- Pygame
- CARLA (planned)

## 📚 Datasets Used

- **CU Lane Dataset** for lane segmentation
- **LISA Traffic Sign Dataset** for sign classification
- **DLDT / LISA** for traffic light classification & detection
- **Mapillary** for sign detection
- **BDD** for vehicle and pedestrian detection

## 📊 Results

| Model        | Task                               | Accuracy / IoU | Dataset   |    Size    | Epochs   |
|--------------|------------------------------------|----------------|-----------|------------|----------|
| CNN          | Sign Classification                | 89%            | GTRSB     |            |20        |
| YOLO       | Sign Detection                     | 89%            | Mapillary |            |50        |
| YOLO       | Traffic light Light Detection      | mAP x          |           |            |50        |
| SCNN         | Lane Clasification                     | IoU x          | Culane    |            |x         |
| CV         | Lane Detection                     | x          | N/A    |            |x         |
| YOLO         | Vehicle & Pedestrian detection     | IoU x          | BDD       | 100k       |30        |


## 🛣️ Roadmap

- [x] Sign classification (CNN)
- [x] Traffic light classification
- [x] Lane detection (U-Net, SCNN, Hough)
- [ ] Integrate all models into Pygame
- [ ] Complete CARLA test scenario

## 📂 Folder Structure

<details>
  <summary>Click to expand folder structure</summary>

```
self-driving-car-simulation/
├── carla/                        # CARLA simulation scripts, camera callbacks, GUI
├── lane-detection/               # Lane detection (Hough, city/highway)
│   ├── city/
│   └── highway/
├── lane-detection-cnn/           # CNN/SCNN lane detection, model tests
├── traffic_sign/                 # Traffic sign detection/classification
├── traffic-lights/               # Traffic light detection/classification
├── vehicle-pedestrian-detection/ # Vehicle & pedestrian detection
├── models/                       # Pretrained models (YOLO, SCNN, CNN, etc)
├── datasets/                     # All datasets (see below)
│   ├── lane-detection/
│   ├── traffic-light/
│   ├── traffic-sign/
│   └── vehicle-pedestrian/
├── results/                      # Training results, metrics, visualizations
├── images/                       # Sample images, predictions, training data
├── notebooks/                    # Jupyter notebooks (experiments, training)
└── videos/                       # Video clips for testing/demo
```

</details>

## ⚡ Quickstart & Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run a demo:**
   ```bash
   python carla/carla_sim.py
   ```
3. **Train a model:**
   See notebooks or scripts in each module folder.

## 📝 Setup & Installation
- Python 3.8+
- See `requirements.txt` for all dependencies
- Optional: CARLA simulator for advanced testing

## 🧠 Model Details
- **Lane Detection:** Hough Transform, SCNN (lane-detection-cnn/)
- **Traffic Sign:** CNN classifier, YOLOv8 detector (traffic_sign/)
- **Traffic Light:** YOLOv8 detector, classifier (traffic-lights/)
- **Vehicle/Pedestrian:** YOLOv8, SCNN (vehicle-pedestrian-detection/)

## 📚 Datasets & Sources
- **Lane Detection:**
  - CU Lane Dataset (`datasets/lane-detection/`)
  - Processed Culane with sorted masks, images, and annotations (`lane-detection/processed/`, Raw Dataset `lane-detection/raw/`)
- **Traffic Sign Classification:**
  - GTSRB Dataset
- **Traffic Sign Detection:**
  - Unprocessed Mapillary Sign Dataset (`datasets/traffic-sign/raw`)
  - Processed dataset for yolov8 format (`datasets/traffic-sign/processed-yolo/`)
- **Traffic Light Detection & Classification:**
  - Unprocessed DLDT & LISA Datasets (`datasets/traffic-light/raw`)
  - Combined DLDT & LISA datasets sorted by light state(`datasets/traffic-light/processed/merged_dataset`)
  - Combined Dataset processed for YOLO training(`datasets/traffic-light/processed/yolo_dataset`)
- **Vehicle & Pedestrian Detection:**
  - BDD100K (Not in repo due to size, can be found on kaggle profile) (`datasets/vehicle-pedestrian/`)
- **Debug Visualizations:**
  - Traffic light debug visualizations (`datasets/traffic-light/debug_visualizations/`)
  - Results visualizations (`results/traffic-sign-classification/visualizations/`, `results/vehicle-pedestrian/visualizations/`)

## 📊 Results
- All training results, metrics, and visualizations are in `results/`
- Example:
  - `results/traffic-sign-classification/metrics/` (JSON, curves)
  - `results/traffic-sign-detection/weights/` (YOLO checkpoints)
  - `results/vehicle-pedestrian/visualizations/` (confusion matrices, sample batches)

## 🚀 Roadmap
- [x] Sign classification (CNN)
- [x] Traffic light classification
- [x] Lane detection (U-Net, SCNN, Hough)
- [ ] Integrate all models into Pygame
- [ ] Complete CARLA test scenario
- [ ] SCNN lane model integration
- [ ] Evaluation scripts for all modules
- [ ] Documentation improvements

## 🤝 Contributing
- Pull requests welcome!
- Please open issues for bugs, feature requests, or questions.

## 🙏 Credits
- Datasets: CU Lane, LISA, GTRSB, Mapillary, BDD100K
- Models: Ultralytics YOLOv8, SCNN, custom CNNs
- Simulation: CARLA, Pygame

## 📅 Changelog
- Aug 2025: Major README overhaul, new folder structure, improved lane detection, robust GUI, generator-based training
- May-Jul 2025: Model improvements, new datasets, CARLA integration
