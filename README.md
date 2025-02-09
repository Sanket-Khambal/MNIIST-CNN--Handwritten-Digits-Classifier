### **Project Name:** MNIST-CNN: Handwritten Digit Classifier  

### **README.md**  

# MNIST-CNN: Handwritten Digit Classifier  

## **Overview**  
MNIST-CNN is a deep learning project that implements a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset**. The goal is to achieve high accuracy in digit recognition while demonstrating the effectiveness of CNNs for image classification. The project is built using **PyTorch** and employs **multiple convolutional layers, pooling layers, and fully connected layers** for feature extraction and classification.  

## **Motivation**  
Handwritten digit classification is a foundational deep learning problem with applications in **digit recognition, postal automation, and banking systems**. This project aims to:  
- Build a CNN to accurately classify digits from the MNIST dataset.  
- Optimize hyperparameters to improve learning efficiency.  
- Compare performance with baseline models (fully connected networks, shallower CNNs).  

## **Features**  
✔️ **CNN architecture** with multiple convolutional and pooling layers.  
✔️ **MNIST dataset preprocessing** (normalization, train-validation split).  
✔️ **Training with PyTorch**, tracking loss and accuracy.  
✔️ **Visualization of model performance**, including learning curves and sample predictions.  

## **Technologies Used**  
- **Python** (NumPy, Matplotlib)  
- **PyTorch** for deep learning  
- **Torchvision** for dataset handling  
- **Matplotlib** for visualization  

## **Model Architecture**  
The CNN consists of the following layers:  
1️⃣ **Conv Layer 1:** 32 filters (3x3 kernel) + ReLU  
2️⃣ **Conv Layer 2:** 64 filters (3x3 kernel) + ReLU  
3️⃣ **MaxPooling:** 2x2 pooling to downsample feature maps  
4️⃣ **Fully Connected Layers:** 128 neurons → 10 output classes (digits 0-9)  
5️⃣ **Softmax Activation:** Final classification layer  

## **Installation & Setup**  
### **1. Install Dependencies**  
```bash
pip install torch torchvision numpy matplotlib
```
### **2. Run Training**  
```bash
python train.py  # Trains the CNN model on the MNIST dataset
```
### **3. Run Evaluation**  
```bash
python evaluate.py  # Tests the model on the MNIST test dataset
```

## **Results & Analysis**  
- Achieved **99.23% test accuracy** after training for **20 epochs**.  
- CNN outperformed fully connected networks and shallow architectures.  
- Model generalizes well with **minimal overfitting**, as validation accuracy closely follows training accuracy.  

