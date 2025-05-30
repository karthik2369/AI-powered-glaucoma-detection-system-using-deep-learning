# AI-powered-glaucoma-detection-system-using-deep-learning
Develop a deep learning-based model (MobileNetV2) for detecting pediatric glaucoma from fundus &amp; OCT images.Incorporate transfer learning with pre-trained models (VGG16, ResNet50) to manage limited data. Use image augmentation to enhance generalization and model robustness.Deploy the trained model into a web-based platform for real-time diagnosis.


# 🔬 Deep Learning-Based Glaucoma Detection System

A deep learning system to detect pediatric glaucoma from retinal and OCT images using MobileNetV2 and transfer learning. Deployed as a web-based interface for early diagnosis in low-resource clinical environments.

## 🚀 Features
- Achieved 96% accuracy and 0.96 F1-score using MobileNetV2.
- Image preprocessing and augmentation with TensorFlow for improved performance.
- Real-time prediction interface using Flask and JavaScript.
- Lightweight model deployable on low-resource systems.

## 🧰 Tech Stack
- Python, TensorFlow, Keras, NumPy, OpenCV
- Flask (for backend)
- HTML, CSS, JavaScript (for frontend)

## 📁 Folder Structure
glaucoma/
├── app.py                  # Flask app
├── model/                  # Contains MobileNetV2 model (.h5)
├── static/                 # Static files (CSS, JS, images)
├── templates/              # HTML files
├── utils.py                # Image preprocessing utilities
├── requirements.txt        # Required libraries
├── README.md               # This file
