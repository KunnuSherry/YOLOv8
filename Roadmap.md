| Step | Title                     | What You Learn                        |
| ---- | ------------------------- | ------------------------------------- |
| 1    | Object Detection Basics   | Understand what detection is          |
| 2    | Learn YOLO Concept        | What YOLO does and how                |
| 3    | Install YOLOv8            | Install and test setup                |
| 4    | Use Pretrained YOLO       | Run YOLO on sample image              |
| 5    | Make Custom Dataset       | Use Roboflow or LabelImg to annotate  |
| 6    | Train on Your Dataset     | Fine-tune model on ambulance images   |
| 7    | Test the Model            | Use the trained model to predict      |
| 8    | Evaluate Model            | Check performance metrics             |
| 9    | Real-World Use            | Deploy or use in live video           |
| 10   | Explore Advanced Features | Try segmentation, exporting, tracking |



# 🧠 YOLOv8 Learning Roadmap (from Scratch)

A complete beginner-friendly step-by-step roadmap to master YOLOv8 using Python and Ultralytics.

---

## ✅ Step 1: Object Detection Basics

**Subtopics:**
- What is Object Detection?
- Bounding Boxes & Class Labels
- IoU (Intersection over Union)
- Precision, Recall, and mAP
- Difference between Classification, Detection, Segmentation

**Resources:**
- [V7 Labs – Object Detection Guide](https://www.v7labs.com/blog/object-detection-guide)
- [Roboflow – What is Object Detection](https://blog.roboflow.com/what-is-object-detection/)
- [YouTube – Object Detection Explained](https://www.youtube.com/watch?v=8Lpp8Qrk-3M)

---

## ✅ Step 2: Understand YOLO (You Only Look Once)

**Subtopics:**
- YOLO Concept and Working
- YOLO Grid and Output Tensor
- Evolution: YOLOv1 → YOLOv8
- YOLOv8: Tasks = detect, segment, classify

**Resources:**
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [Roboflow – YOLOv8 Guide](https://blog.roboflow.com/yolov8-guide/)
- [YouTube – YOLO Explained](https://www.youtube.com/watch?v=Grir6TZbc1M)

---

## ✅ Step 3: Install YOLOv8

**Subtopics:**
- Python 3.8+ installation
- Create a virtual environment
- Install Ultralytics
- Test YOLO CLI

**Commands:**
```bash
pip install ultralytics
yolo
```

**Resources:**
- [Ultralytics Install Guide](https://docs.ultralytics.com/)
- [YouTube – Install YOLOv8](https://www.youtube.com/watch?v=J3f54b-J5uA)

---

## ✅ Step 4: Use Pretrained YOLOv8 Model

**Subtopics:**
- Understand model sizes: n, s, m, l, x
- Predict on image
- Predict on webcam

**Commands:**
```bash
yolo task=detect mode=predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg"
```

**Resources:**
- [Predict Mode Docs](https://docs.ultralytics.com/modes/predict/)
- [YouTube – Run Pretrained YOLOv8](https://www.youtube.com/watch?v=H2jK5JuVIQQ)

---

## ✅ Step 5: Create Your Custom Dataset

**Subtopics:**
- Collect images (Google, Roboflow, etc.)
- Label images (LabelImg or Roboflow)
- YOLO format labels
- `data.yaml` file structure

**Example `data.yaml`:**
```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: ['ambulance']
```

**Resources:**
- [Roboflow](https://roboflow.com)
- [LabelImg](https://github.com/tzutalin/labelImg)
- [Dataset Format Guide](https://docs.ultralytics.com/datasets/)

---

## ✅ Step 6: Train a Pretrained YOLOv8 Model (Transfer Learning)

**Subtopics:**
- Load pretrained weights (e.g., `yolov8n.pt`)
- Fine-tune on your dataset
- Monitor training logs and results

**Commands:**
```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=50 imgsz=640
```

**Resources:**
- [Train Mode Docs](https://docs.ultralytics.com/modes/train/)
- [YouTube – Custom YOLOv8 Training](https://www.youtube.com/watch?v=EzqTr1g5vdk)

---

## ✅ Step 7: Test Your Trained Model

**Subtopics:**
- Load best weights
- Test on images, webcam, or videos

**Commands:**
```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source="test.jpg"
yolo task=detect mode=predict model=best.pt source=0
```

**Resources:**
- [Predict Docs](https://docs.ultralytics.com/modes/predict/)
- [YouTube – Testing Trained Model](https://www.youtube.com/watch?v=fa9z0_M5FXA)

---

## ✅ Step 8: Evaluate the Model

**Subtopics:**
- Run validation
- Check metrics (Precision, Recall, mAP)
- Analyze confusion matrix

**Command:**
```bash
yolo task=detect mode=val model=best.pt data=dataset/data.yaml
```

**Resources:**
- [Validation Docs](https://docs.ultralytics.com/modes/val/)
- [mAP Explanation – YouTube](https://www.youtube.com/watch?v=FppGJoC_L1U)

---

## ✅ Step 9: Deploy in Real World

**Subtopics:**
- Run live detection on webcam
- RTSP/Video input
- Export model to ONNX/CoreML/TensorRT
- Use in Flask app or Raspberry Pi

**Command:**
```bash
yolo export model=best.pt format=onnx
```

**Resources:**
- [Exporting YOLOv8](https://docs.ultralytics.com/modes/export/)
- [YOLOv8 with Flask](https://www.youtube.com/watch?v=x2kkrHQI7zI)

---

## ✅ Step 10: Explore More YOLOv8 Tasks

**Subtopics:**
- Segmentation: `task=segment`
- Classification: `task=classify`
- Tracking using DeepSORT
- Data Augmentation & Hyperparameters

**Resources:**
- [YOLOv8 Segmentation Guide](https://docs.ultralytics.com/tasks/segment/)
- [WandB + YOLO](https://docs.wandb.ai/guides/integrations/ultralytics)
- [YouTube – YOLOv8 DeepSORT](https://www.youtube.com/watch?v=o2s_R5y4FLI)

---

# 🧠 Bonus: Pretrained Model Training Summary

To **fine-tune a pretrained YOLOv8 model**, use:
```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=50 imgsz=640
```

To **train from scratch** (no pretrained weights), use:
```bash
yolo task=detect mode=train model=yolov8n.yaml data=dataset/data.yaml epochs=100 imgsz=640
```

---

Happy Learning! 🚀  
Made for: **Kunal Sharma**
