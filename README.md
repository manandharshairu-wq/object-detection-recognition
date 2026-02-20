# object-detection-recognition

Object Detection and Recognition

This project implements and compares two modern object detection architectures under limited data and 8 GB GPU constraints. The goal is to analyze the trade-off between detection accuracy and computational efficiency using both a two-stage detector (Faster R-CNN) and a one-stage detector (YOLOv8n).

The experiments are conducted on:

Penn-Fudan Pedestrian Dataset (single-class detection)

Oxford-IIIT Pets Subset (multi-class fine-grained detection)

🚀 Models Used

Faster R-CNN (MobileNetV3 FPN)

Two-stage detector

Strong localization accuracy

Higher computational cost

YOLOv8n (Nano)

One-stage detector

Optimized for speed and efficiency

Suitable for low-memory GPUs

📁 Repository Structure
object-detection-recognition/
│
├── src/              # Training and utility scripts
├── splits/           # Train/val/test splits
├── outputs/          # Faster R-CNN outputs
├── runs/detect/      # YOLO training logs and outputs
├── requirements.txt
├── Report_Assignment2.pdf
└── README.md
⚙️ Environment Setup
1️⃣ Clone the repository
git clone https://github.com/<your-username>/object-detection-recognition.git
cd object-detection-recognition
2️⃣ Install dependencies
pip install -r requirements.txt

Recommended:

Python ≥ 3.9

PyTorch with CUDA support (for GPU training)

NVIDIA GPU with ~8 GB VRAM

▶️ How to Run

Run the pipeline in the following order.

✅ Step 1 — Initialize environment

First run the setup/runner logic inside src/ to:

set random seeds

load dataset splits

configure training parameters

initialize models

⚠️ This step must be executed before training.

✅ Step 2 — Train Faster R-CNN

Run the Faster R-CNN training script from src/.

This stage:

loads pretrained MobileNetV3 FPN

fine-tunes on Penn-Fudan

fine-tunes on Pets subset

computes precision, recall, and mAP

saves qualitative predictions

Outputs saved to:

outputs/
✅ Step 3 — Train YOLOv8n

Run the YOLO training pipeline.

This stage:

loads YOLOv8n nano weights

trains on both datasets

evaluates detection metrics

generates training curves

Logs and outputs saved to:

runs/detect/
✅ Step 4 — Review results

After training completes, the pipeline:

aggregates metrics

saves plots and curves

exports qualitative detections

prepares figures used in the report

📊 Evaluation Metrics

Models are evaluated using:

mAP@0.5

Precision

Recall

Training time

Inference speed

A detection is considered correct when IoU ≥ 0.5 between predicted and ground-truth boxes.

📈 Key Findings

Faster R-CNN achieves slightly higher localization accuracy

YOLOv8n provides significantly faster inference

Both models perform near saturation on Penn-Fudan

Performance differences become clearer on the multi-class Pets task

Model choice depends on the accuracy–speed requirements of deployment

⚠️ Notes

Large datasets and checkpoints are not included due to size limits

Results may vary slightly depending on GPU and random seed

Mixed precision (FP16) is used when available

Training should be run sequentially

📄 Report

See Report_Assignment2.pdf for full experimental details, analysis, and discussion.

👩‍💻 Author

Shaira Manandhar
Beloit College — Data Science & Business Management


