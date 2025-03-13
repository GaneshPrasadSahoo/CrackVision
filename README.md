# CrackVision

**CrackVision** is an AI-powered crack detection system that classifies surfaces as **"Crack"** or **"Not Crack"** using deep learning. The project utilizes two pre-trained models, **VGG16** and **ResNet50**, with **ResNet50 achieving 98% accuracy**, making it the preferred model for predictions.

## 🚀 Features

- **Deep Learning-Based Detection** – Uses AI to accurately classify cracks.
- **High Accuracy** – ResNet50 model achieves 98% accuracy.
- **User-Friendly Interface** – Simple image upload and classification via Streamlit.
- **Real-Time Analysis** – Provides instant predictions.
- **Confidence Score** – Displays model certainty for each classification.

## 📂 Dataset

The model was trained on a dataset containing images of cracked and non-cracked surfaces. The dataset was preprocessed to enhance model performance and accuracy.

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **ResNet50 (Pre-trained Model)**
- **VGG16 (For comparison)**
- **NumPy & OpenCV**
- **Streamlit (For UI)**

## 🔧 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/GaneshPrasadSahoo/CrackVision.git
   cd CrackVision
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 📌 How to Use

1. Upload an image of a surface.
2. Click **Predict** to classify it as **Crack** or **Not Crack**.
3. View the prediction result along with the confidence score.

## 🔥 Future Enhancements

- Integrate real-time video crack detection.
- Expand dataset for better generalization.
- Develop a web API for external integrations.

## 🤝 Contributing

Contributions are welcome! Fork this project, make your improvements, and submit a pull request.

## 📜 License

This project is licensed under the **MIT License**.

---

### 👨‍💻 Developed by *Ganesh Prasad Sahoo*

🚀 CrackVision - AI for Safer Structures!

🔗 [GitHub Repository](https://github.com/GaneshPrasadSahoo/CrackVision)

