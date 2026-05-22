# PyTorch OpenCV Projects

This repository contains a collection of computer vision and deep learning projects developed using **PyTorch**, **OpenCV**, **ONNX**, and **TensorFlow Lite**.

The projects cover important topics such as image classification, object localization, image enhancement, tabular data processing, model compression, and model conversion for deployment.

## Overview

Computer vision is an important field in artificial intelligence that allows machines to understand and process visual information from images and videos.

This repository includes several practical projects that demonstrate the use of deep learning and image processing techniques for solving vision-related tasks.

The main goal of this repository is to build a strong foundation in:

- Image classification
- Object localization
- Image enhancement
- Model compression
- Model deployment
- ONNX and TensorFlow Lite conversion

## Projects Included

| Project | Description |
|---|---|
| Image Classification | Trains and evaluates a deep learning model for classifying images into different categories |
| Object Localization | Predicts both the object class and its location inside an image |
| Basic Image Enhancement | Applies mathematical operations for improving image quality |
| Tabular Data | Demonstrates data processing and machine learning on structured data |
| Model Compression | Explores model optimization and compression techniques |
| ONNX / TFLite Conversion | Converts trained models into deployment-friendly formats |

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook
- ONNX
- TensorFlow Lite

## Project Structure

```text
PyTorchOpenCVProjects/
│
├── 04_Basic_Image_Enhancement_Mathematical_Operations.ipynb
│   └── Image enhancement using mathematical operations
│
├── Image_Classification.ipynb
│   └── Deep learning based image classification project
│
├── Object Localization.ipynb
│   └── Object localization using deep learning
│
├── Tabular_data.ipynb
│   └── Tabular data processing and machine learning experiment
│
├── compression_project.py
│   └── Model compression and optimization script
│
├── baseline_model.pt
│   └── Saved PyTorch baseline model
│
├── model.onnx
│   └── Exported ONNX model
│
├── model_quant.onnx
│   └── Quantized ONNX model
│
├── model.tflite
│   └── TensorFlow Lite model for lightweight deployment
│
├── onnx_to_tflite.py
│   └── Script for converting ONNX model to TensorFlow Lite
│
├── onnx_to_lite/
│   └── Model conversion related files
│
├── onnx_to_lflite/
│   └── Model conversion related files
│
├── onnx_to_tflite/
│   └── Model conversion related files
│
└── README.md
    └── Project documentation
```

## Main Project Descriptions

### 1. Image Classification

The image classification project focuses on training a deep learning model to classify images into different categories.

This project demonstrates:

- Loading image datasets
- Preprocessing images
- Building a PyTorch model
- Training and validation
- Evaluating model performance

### 2. Object Localization

Object localization is a computer vision task where the model predicts not only the object class but also the position of the object in the image.

This project demonstrates:

- Image-based prediction
- Bounding box regression
- Object position estimation
- Deep learning model training for localization tasks

### 3. Basic Image Enhancement

This notebook explores basic image processing operations using mathematical techniques.

It includes concepts such as:

- Pixel-level operations
- Brightness and contrast adjustment
- Image transformation
- Basic OpenCV processing

### 4. Tabular Data

This notebook contains experiments with structured data. It helps demonstrate basic machine learning workflow outside image-based data.

It includes:

- Data loading
- Data preprocessing
- Feature handling
- Model training workflow

### 5. Model Compression

The model compression project focuses on reducing model size and improving deployment efficiency.

This is useful when deploying deep learning models on devices with limited computational resources.

### 6. Model Conversion

This repository also includes ONNX and TensorFlow Lite model files. These formats are useful for deploying models outside the normal PyTorch training environment.

The conversion workflow can be represented as:

```text
PyTorch model
      ↓
ONNX model
      ↓
Quantized ONNX model
      ↓
TensorFlow Lite model
      ↓
Deployment on lightweight devices
```

## How to Run

First, clone the repository:

```bash
git clone https://github.com/PrithviRaj037/PyTorchOpenCVProjects.git
```

Go into the project folder:

```bash
cd PyTorchOpenCVProjects
```

Install the required Python libraries:

```bash
pip install torch torchvision opencv-python numpy matplotlib onnx onnxruntime tensorflow
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open any notebook, for example:

```text
Image_Classification.ipynb
```

or

```text
Object Localization.ipynb
```

## Example Workflow

```text
Input image
    ↓
Image preprocessing
    ↓
Deep learning model
    ↓
Prediction
    ↓
Classification / localization result
    ↓
Model export for deployment
```

## Skills Demonstrated

This repository demonstrates practical skills in:

- Deep learning
- Computer vision
- Image classification
- Object localization
- Image processing
- Model optimization
- Model conversion
- PyTorch model development
- OpenCV-based image processing
- Deployment-oriented AI workflows

## Applications

The techniques in this repository are relevant for:

- Autonomous driving perception
- Robotics vision
- Object detection and localization
- Medical image analysis
- Industrial inspection
- Edge AI deployment
- Real-time computer vision systems

## Future Improvements

Possible improvements for this repository include:

- Add example output images for each project
- Add a `requirements.txt` file
- Organize each project into separate folders
- Add training results and accuracy scores
- Add explanation of datasets used
- Add model architecture diagrams
- Add inference scripts
- Add comparison between original and compressed models
- Add deployment instructions for ONNX Runtime and TensorFlow Lite

## Recommended Future Repository Structure

For better readability, this repository can be reorganized like this:

```text
PyTorchOpenCVProjects/
│
├── image_classification/
│   ├── Image_Classification.ipynb
│   ├── README.md
│   └── results/
│
├── object_localization/
│   ├── Object_Localization.ipynb
│   ├── README.md
│   └── results/
│
├── image_enhancement/
│   ├── Basic_Image_Enhancement.ipynb
│   └── README.md
│
├── tabular_data/
│   ├── Tabular_data.ipynb
│   └── README.md
│
├── model_compression/
│   ├── compression_project.py
│   ├── baseline_model.pt
│   ├── model.onnx
│   ├── model_quant.onnx
│   ├── model.tflite
│   └── README.md
│
├── requirements.txt
└── README.md
```

## Author

**Prithvi Raj**

GitHub: [PrithviRaj037](https://github.com/PrithviRaj037)

## License

This project is intended for educational and research purposes.
