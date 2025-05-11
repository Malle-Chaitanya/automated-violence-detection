# 🔍 Automated Violence Detection System

This project is a deep learning-based system designed to automatically detect violence in video content using frame-based classification. It uses a fine-tuned **MobileNetV2** model and provides a user-friendly **Streamlit** web application for real-time or uploaded video analysis.

---

## 📌 Features

- 🎥 Real-time violence detection via webcam
- 📁 Upload video files for analysis
- ⚠️ Violence probability percentage shown on screen
- 🚸 Kid safety warning if violence > 60%
- 📊 Streamlit-based interactive interface
- 💾 Pre-trained model included

---

## 🧠 Model Details

- **Model Architecture:** MobileNetV2
- **Training Type:** Frame-based classification on violent/non-violent video clips
- **Libraries Used:** TensorFlow, Keras, OpenCV, NumPy, Streamlit

---

## 📂 Project Structure

📁 automated-violence-detection/
│
├── app.py # Streamlit web app
├── modelnew.h5 # Trained MobileNetV2 model
├── ModelWeights.weights.h5 # Custom weights
├── *.ipynb # Jupyter notebooks for training/evaluation
├── archive/ # Dataset directory
├── logs/ # TensorBoard logs (optional)
├── accuracy.png / loss.png # Training visualizations
└── requirements.txt # Required Python packages

2️⃣ Set Up Environment
(Optional but recommended)

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


3️⃣ Install Dependencies

pip install -r requirements.txt
Or manually:

pip install streamlit opencv-python tensorflow keras numpy matplotlib


4️⃣ Run the App

streamlit run app.py


🖼️ Output Preview

Live detection results with frame overlay
Probability score of violence shown in real-time
Kid-safe alert if threshold > 60%

📌 Notes

Ensure you have ModelWeights.weights.h5 in the root directory.
The .ipynb_checkpoints/ and archive/ folder contains training data and can be optionally excluded in deployment.


