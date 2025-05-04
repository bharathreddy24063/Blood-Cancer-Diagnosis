# Blood-Cancer-Diagnosis
# 🧬 Cancer Detection and Segmentation System (VGG16 + U-Net)

A deep learning-based web application to detect and classify cancer from blood smear images using a hybrid **VGG16 + U-Net** model. The system provides both **classification** (Benign, Malignant-Early, Malignant-Pre, Malignant-Pro) and **segmentation** to highlight cancer-affected regions. It supports **Doctor and Admin dashboards**, patient record management, and report generation.

---

## 📌 Features

- 🔐 User authentication (Admin & Doctor roles)
- 📤 Image upload for cancer diagnosis
- 🧠 VGG16-based cancer classification
- 🧬 U-Net-based segmentation of affected areas
- 📊 Report generation with results and confidence scores
- 🗂️ Patient record management
- 🌐 Flask-based web interface

---

## 🚀 Tech Stack

| Layer          | Technology Used                          |
|----------------|-------------------------------------------|
| Frontend       | HTML5, CSS3, JavaScript                   |
| Backend        | Python, Flask                             |
| Deep Learning  | TensorFlow / Keras (VGG16 + U-Net)        |
| Image Handling | OpenCV, PIL                               |
| Database       | MySQL                           |
| Reporting      | ReportLab (for PDF generation)            |

---

## 📁 Project Structure
project-root/
│
├── static/ # Static files (CSS, JS, images)
├── templates/ # HTML templates (admin.html, doctor.html, login.html, etc.)
├── uploads/ # Uploaded images
├── model/
│ ├── vgg16_unet_model.h5 # Trained model
│
│
├── app.py # Main Flask application
├── database.db # SQLite database (optional)
├── requirements.txt # Python dependencies
└── README.md # Project documentation




---

## 🧪 Model Overview

- **VGG16**: Used as the base for feature extraction and classification.
- **U-Net**: Used for pixel-level segmentation to detect cancer regions.
- **Input**: RGB Blood smear images resized to 256x256.
- **Output**: Cancer type + cancer region mask.

### Classes:

- **Benign**
- **Malignant - Early**
- **Malignant - Pre**
- **Malignant - Pro**

---




