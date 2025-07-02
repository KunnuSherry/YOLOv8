# 🚀 YOLOv8 Custom Object Detection with Roboflow

A complete guide to train your own YOLOv8 object detection model using Roboflow for dataset management and Google Colab for training.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Dataset Creation with Roboflow](#step-1-dataset-creation-with-roboflow)
- [Step 2: Image Upload and Annotation](#step-2-image-upload-and-annotation)
- [Step 3: Dataset Export](#step-3-dataset-export)
- [Step 4: Google Colab Setup](#step-4-google-colab-setup)
- [Step 5: Install Dependencies](#step-5-install-dependencies)
- [Step 6: Model Training](#step-6-model-training)
- [Step 7: Model Prediction](#step-7-model-prediction)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## 🔧 Prerequisites

- Google account (for Colab access)
- Roboflow account (free tier available)
- Collection of images for your custom object detection task
- Basic understanding of machine learning concepts

## 📊 Step 1: Dataset Creation with Roboflow

1. **Sign up/Login to Roboflow**
   - Visit [roboflow.com](https://roboflow.com)
   - Create a free account or login to existing account

2. **Create New Project**
   ```
   - Click "Create New Project"
   - Choose "Object Detection" as project type
   - Give your project a meaningful name
   - Select appropriate license (Public Domain for learning)
   ```

3. **Project Configuration**
   - Set up your class names (objects you want to detect)
   - Configure project settings as needed

## 📷 Step 2: Image Upload and Annotation

### Upload Images
1. **Drag and Drop Images**
   - Select "Upload" from the project dashboard
   - Drag and drop your images or browse to select
   - Recommended: 100-1000+ images for good performance

2. **Image Requirements**
   - High quality images (at least 416x416 pixels)
   - Diverse angles, lighting conditions, and backgrounds
   - Include edge cases and difficult scenarios

### Annotate Images
1. **Start Annotating**
   - Click on "Annotate" tab
   - Select an image to begin annotation

2. **Create Bounding Boxes**
   ```
   - Click and drag to create bounding boxes around objects
   - Assign correct class labels to each box
   - Be precise with box boundaries
   - Ensure all instances of target objects are labeled
   ```

3. **Annotation Best Practices**
   - Tight bounding boxes (close to object edges)
   - Consistent labeling across all images
   - Label all instances, even partially occluded objects
   - Double-check annotations for accuracy

## 📦 Step 3: Dataset Export

1. **Generate Dataset Version**
   ```
   - Navigate to "Generate" tab
   - Click "Create Version"
   - Apply preprocessing (optional):
     * Auto-Orient
     * Resize (640x640 recommended for YOLOv8)
   - Apply augmentations (optional):
     * Rotation, flip, brightness adjustment
     * Recommended: 2-3x augmentation for small datasets
   ```

2. **Export Dataset**
   ```
   - Click "Export Dataset"
   - Format: Select "YOLOv8"
   - Download method: Choose "download zip" or "show download code"
   - Copy the download code for use in Colab
   ```

## 💻 Step 4: Google Colab Setup

1. **Open Google Colab**
   - Visit [colab.research.google.com](https://colab.research.google.com)
   - Create new notebook or open existing one

2. **Enable GPU (Recommended)**
   ```
   Runtime → Change runtime type → Hardware accelerator → GPU (T4/V100)
   ```

3. **Verify GPU Access**
   ```python
   !nvidia-smi
   ```

## 📚 Step 5: Install Dependencies

```python
# Install required packages
!pip install ultralytics roboflow

# Import necessary libraries
from ultralytics import YOLO
from roboflow import Roboflow
import os
```

### Download Dataset from Roboflow
```python
# Initialize Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")  # Get this from your Roboflow account
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT_NAME")
dataset = project.version(1).download("yolov8")
```

## 🏋️ Step 6: Model Training

### Basic Training Configuration
```python
# Load a YOLOv8 model
model = YOLO('yolov8n.pt')  # nano version (fastest)
# Alternative options: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Train the model
results = model.train(
    data=f"{dataset.location}/data.yaml",  # path to dataset YAML
    epochs=100,                            # number of training epochs
    imgsz=640,                            # input image size
    batch=16,                             # batch size
    mode='train'                          # training mode
)
```

### Advanced Training Configuration
```python
# Advanced training with custom parameters
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,                    # initial learning rate
    weight_decay=0.0005,         # weight decay
    warmup_epochs=3,             # warmup epochs
    box=7.5,                     # box loss weight
    cls=0.5,                     # class loss weight
    dfl=1.5,                     # distribution focal loss weight
    save_period=10,              # save model every N epochs
    mode='train'
)
```

### Monitor Training
```python
# View training results
from IPython.display import Image, display
display(Image('runs/detect/train/results.png'))
display(Image('runs/detect/train/confusion_matrix.png'))
```

## 🔮 Step 7: Model Prediction

### Load Trained Model
```python
# Load your trained model
model = YOLO('runs/detect/train/weights/best.pt')
```

### Single Image Prediction
```python
# Predict on a single image
results = model.predict(
    source='path/to/your/image.jpg',
    mode='predict',
    save=True,                    # save prediction images
    conf=0.5,                     # confidence threshold
    iou=0.45                      # IoU threshold for NMS
)

# Display results
for result in results:
    result.show()
```

### Batch Prediction
```python
# Predict on multiple images
results = model.predict(
    source='path/to/images/folder/',
    mode='predict',
    save=True,
    conf=0.5,
    save_txt=True,               # save predictions as txt files
    save_conf=True               # save confidence scores
)
```

### Video Prediction
```python
# Predict on video
results = model.predict(
    source='path/to/video.mp4',
    mode='predict',
    save=True,
    stream=True                  # stream results for memory efficiency
)
```

### Real-time Webcam Prediction
```python
# Real-time prediction from webcam
results = model.predict(
    source=0,                    # 0 for default webcam
    mode='predict',
    show=True,                   # display results in real-time
    stream=True
)
```

## 🔧 Troubleshooting

### Common Issues and Solutions

**Issue: CUDA out of memory**
```python
# Solution: Reduce batch size
model.train(data="data.yaml", epochs=100, batch=8)  # Reduce from 16 to 8
```

**Issue: Low mAP scores**
```python
# Solutions:
# 1. Increase training epochs
# 2. Add more diverse training data
# 3. Increase image resolution
# 4. Adjust learning rate
model.train(data="data.yaml", epochs=200, imgsz=832, lr0=0.001)
```

**Issue: Dataset not found**
```python
# Check dataset path
import os
print(f"Dataset location: {dataset.location}")
print(f"Files in dataset: {os.listdir(dataset.location)}")
```

### Performance Optimization Tips

1. **Data Quality**: Ensure high-quality, diverse annotations
2. **Augmentation**: Use appropriate data augmentation
3. **Model Size**: Balance between speed and accuracy
4. **Hyperparameters**: Experiment with learning rate and batch size
5. **Training Time**: More epochs generally improve performance

## 📈 Model Evaluation

```python
# Validate model performance
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# Export model for deployment
model.export(format='onnx')  # Export to ONNX format
```

## 🚀 Deployment Options

### Export Formats
```python
# Various export formats
model.export(format='torchscript')  # TorchScript
model.export(format='coreml')       # CoreML (iOS)
model.export(format='tflite')       # TensorFlow Lite
model.export(format='pb')           # TensorFlow SavedModel
```

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [Computer Vision Best Practices](https://blog.roboflow.com/)

## 📝 Notes

- **Training Time**: Expect 1-4 hours depending on dataset size and hardware
- **GPU Recommendation**: Use Google Colab Pro for faster training with better GPUs
- **Dataset Size**: Minimum 100 images per class for decent performance
- **Validation Split**: Roboflow automatically splits data (70% train, 20% valid, 10% test)

## 🎯 Quick Command Reference

```bash
# Essential commands for quick reference
pip install ultralytics roboflow
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"
yolo train data=data.yaml model=yolov8n.pt epochs=100
yolo predict model=best.pt source=image.jpg
```

---
