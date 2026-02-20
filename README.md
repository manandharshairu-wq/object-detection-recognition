# Object Detection and Recognition

This project implements and compares two modern object detection architectures under limited data and 8 GB GPU constraints. The objective is to analyze the trade-off between detection accuracy and computational efficiency using both a two-stage detector (Faster R-CNN) and a one-stage detector (YOLOv8n).

The experiments are conducted on:

- **Penn-Fudan Pedestrian Dataset** (single-class detection)  
- **Oxford-IIIT Pets Subset** (multi-class fine-grained detection)

---

## Models Used

### Faster R-CNN (MobileNetV3 FPN)
- Two-stage detector  
- Strong localization accuracy  
- Higher computational cost  

### YOLOv8n (Nano)
- One-stage detector  
- Optimized for speed and efficiency  
- Suitable for low-memory GPUs  

---

## Repository Structure
object-detection-recognition/
│
├── src/ # Training and utility scripts
├── splits/ # Train/val/test splits
├── outputs/ # Faster R-CNN outputs
├── runs/detect/ # YOLO training logs and outputs
├── requirements.txt
├── Report_Assignment2.pdf
└── README.md


---
<img width="805" height="473" alt="Screenshot 2026-02-20 at 4 59 10 PM" src="https://github.com/user-attachments/assets/0305d08c-56fc-4c26-83bd-4d894f6a8834" />

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/object-detection-recognition.git
cd object-detection-recognition

### 2. Install dependencies

pip install -r requirements.txt

