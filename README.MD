# 🧠 Tumor Detection Model + Grad-CAM Visualization  

This repository contains the **training pipeline** and **explainability module** for tumor detection using deep learning.  
It includes:  
- A trained **PyTorch model** (`disease_model.pth`)  
- **Grad-CAM utility** for applying heatmaps to highlight tumor regions  
- **Deployment-ready scripts** with Docker support  

---

## 🚀 Project Overview
- Train and evaluate deep learning models for **tumor detection**.  
- Use **Grad-CAM** to generate heatmaps over detected tumor regions for explainability.  
- Containerized using **Docker** for easy deployment.  


---

## ⚡ Features
- 🧑‍⚕️ **Tumor Detection Model** (PyTorch)  
- 🔥 **Grad-CAM Heatmaps** for explainability  
- 📦 **Docker Support** for deployment  
- 📜 Lightweight and reproducible environment  

---

## 🛠️ Tech Stack
- **Framework**: PyTorch  
- **Visualization**: Grad-CAM (OpenCV, Matplotlib)  
- **Deployment**: FastAPI +Huggingface

---

## 🖥️ Running Locally
Clone the repository and install dependencies:  

```bash
# Clone repo
git clone https://github.com/manthirapriya18/<repo-name>.git
cd <repo-name>

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py



⚠️ **Medical Disclaimer**: This tool is for educational and research purposes only. Always consult qualified medical professionals for diagnosis and treatment.
