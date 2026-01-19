# 🐶🐱 Dog vs Cat Image Classifier  

Welcome to my **first deep learning project** 🎉  
This project uses **Transfer Learning with VGG16** to classify images of **dogs and cats** using **TensorFlow and Keras**.  
The model is trained on a filtered version of the **Microsoft Cats vs Dogs dataset** and demonstrates the complete deep learning workflow.

---

## 📌 Project Overview

Image classification is a fundamental computer vision task.  
In this project, a **pretrained VGG16 model** is used as a feature extractor, with custom layers added on top to perform **binary classification** (Dog vs Cat).

The project covers:
- Dataset loading and preprocessing  
- Data augmentation  
- Transfer learning  
- Model training and evaluation  
- Visualization of results  
- Image prediction on custom inputs  

---

## 🚀 Features

- ✅ Transfer Learning using **VGG16 (ImageNet weights)**  
- ✅ Data augmentation for improved generalization  
- ✅ Early stopping to prevent overfitting  
- ✅ Model checkpointing (`best_model.h5`)  
- ✅ Training & validation accuracy/loss visualization  
- ✅ Custom image prediction function  
- ✅ Google Colab–compatible image upload support  

---

## 🧠 Model Architecture

- **Base Model**: VGG16 (frozen, pretrained on ImageNet)  
- **Custom Layers**:
  - Global Average Pooling  
  - Batch Normalization  
  - Dense layer (512 units, ReLU)  
  - Dropout (0.5)  
  - Sigmoid output layer (binary classification)  

---

## 📊 Output & Visualizations

### 🔹 Training & Validation Accuracy
This plot shows how the model accuracy improves over training epochs.

![Training Accuracy](https://github.com/prasan23bad042/Cat_vs_Dog_prediction/blob/main/output1-model.png?raw=true)

---

### 🔹 Training & Validation Loss
This plot helps analyze convergence and overfitting behavior.

![Training Loss](https://github.com/prasan23bad042/Cat_vs_Dog_prediction/blob/main/output2-accuracy.png?raw=true)

---

### 🔹 Sample Prediction Output
Prediction result for a custom uploaded image.

![Prediction Output](https://github.com/prasan23bad042/Cat_vs_Dog_prediction/blob/main/output4-own%20image%20check.png?raw=true)

---

## 📈 Model Performance

- Uses **Binary Crossentropy** loss  
- Optimized with **Adam optimizer**  
- Evaluated using:
  - Training accuracy  
  - Validation accuracy  
  - Loss curves  

The model generalizes well due to transfer learning and data augmentation.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/prasan23bad042/Cat_vs_Dog_prediction.git
cd Cat_vs_Dog_prediction
