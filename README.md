# Object Detection and Recognition


This repository compares two lightweight object detection models under limited data and ~8 GB GPU constraints.

## Models

- **Faster R-CNN (MobileNetV3 FPN)** — two-stage detector with strong localization
- **YOLOv8n (Nano)** — one-stage detector optimized for speed and efficiency

## Datasets

- **Penn-Fudan Pedestrian** — single-class pedestrian detection (~170 images)
- **Oxford-IIIT Pets (subset)** — multi-class cat breed detection

---

## Repository Structure
object-detection-recognition/
├── outputs/     # Saved outputs (plots, predictions, metrics)
├── runs/
│ └── detect/   # YOLO training logs and predictions
├── splits/     # Train/val/test split files
├── src/        # Training and utility Python scripts
├── requirements.txt          # Dependencies
├── Report_Assignment2.pdf    # Final technical report
└── README.md


---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/object-detection-recognition.git
cd object-detection-recognition

<img width="805" height="522" alt="Screenshot 2026-02-20 at 5 03 48 PM" src="https://github.com/user-attachments/assets/3fdf2728-fd6b-47d8-a749-963a16b82b8d" />

dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

<img width="805" height="640" alt="Screenshot 2026-02-20 at 5 04 28 PM" src="https://github.com/user-attachments/assets/484c3fb7-637f-483f-ba52-bf0b5465ca2f" />

Outputs are saved to: outputs/

<img width="805" height="253" alt="Screenshot 2026-02-20 at 5 05 29 PM" src="https://github.com/user-attachments/assets/e98cfcde-cc58-4943-bc20-e4958a848f7f" />
Ultralytics are saved to: runs/detect
<img width="805" height="184" alt="Screenshot 2026-02-20 at 5 06 17 PM" src="https://github.com/user-attachments/assets/706b6e1f-8af0-4be2-a7f5-4b7c6e8fd571" />
<img width="805" height="565" alt="Screenshot 2026-02-20 at 5 06 38 PM" src="https://github.com/user-attachments/assets/a7f2c873-f974-471f-8c1e-736348e6a453" />

