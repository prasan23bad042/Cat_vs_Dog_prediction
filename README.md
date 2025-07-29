# 🐶🐱 Dog vs Cat Image Classifier

Welcome to my first deep learning project! 🎉 This project uses **transfer learning** with the **VGG16** model to classify images of **dogs and cats**. It was built using **TensorFlow** and trained on a subset of the official Cats vs Dogs dataset from Microsoft.

---

## 📁 Project Structure

- **Dog vs Cat Prediction.py** – Main training script.
- **best_model.h5** – Saved best model weights (generated after training).
- Dataset – Automatically downloaded and unzipped inside the script.

---

## 🚀 Features

- ✅ Transfer Learning using VGG16
- ✅ Data Augmentation for better generalization
- ✅ Early stopping and model checkpointing
- ✅ Training/validation accuracy and loss plots
- ✅ Simple image prediction function
- ✅ Colab-compatible image upload and testing

---

## 🧠 Model Overview

- **Base Model**: VGG16 (pretrained on ImageNet, frozen during training)
- **Custom Layers**:
  - Global Average Pooling
  - Batch Normalization
  - Dense (512 units) + Dropout
  - Final Sigmoid Layer for Binary Classification

---

## 🏁 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Cat_vs_Dog_prediction.git
cd Cat_vs_Dog_prediction
