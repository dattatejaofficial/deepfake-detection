## Deepfake Detection using EfficientNet-B1

### Overview

This project focuses on detecting **Deepfake and manipulated facial images** using a **Convolutional Neural Network (CNN)** based on **EfficientNet-B1**. It provides an end-to-end pipeline — from training and evaluation to real-time inference through a **Flask web application**.

---

### Features

* Deepfake image classification using **EfficientNet-B1**
* Ready-to-use **Flask web interface** for uploading and predicting images
* REST API for programmatic predictions
* Pre-trained model (`model.keras`) for immediate inference
* Jupyter notebooks for transparent model training and testing

---

### Project Structure

```
deepfake-detection-main/
│
├── Deepfake Training with EfficientNet B1.ipynb   # Model training notebook
├── Model Testing.ipynb                            # Model evaluation and testing
├── app.py                                         # Flask web application
├── api.py                                         # REST API for predictions
├── api testing.py                                 # API endpoint testing
├── model.keras                                    # Trained EfficientNet-B1 model
├── requirements.txt                               # Python dependencies
├── templates/
│   ├── home.html                                  # Upload page
│   └── prediction.html                            # Result display page
└── .gitignore
```

---

### Dataset

This project uses the **[Real and Fake Face Detection Dataset](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)** by **CIPL Lab** on Kaggle.

| Property          | Description                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Dataset Name**  | Real and Fake Face Detection                                                                                        |
| **Source**        | [Kaggle – ciplab/real-and-fake-face-detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) |
| **Total Images**  | 1,080 Real + 960 Fake                                                                                               |
| **Format**        | `.jpg`                                                                                                              |
| **Preprocessing** | Images resized to **224×224 pixels**, normalized to [0,1] range                                                     |
| **Split Ratio**   | 70% Training, 20% Validation, 10% Testing                                                                           |
| **Goal**          | Classify whether an image of a face is **Real** or **Fake**                                                         |

---

### ⚙️ Installation & Setup

1. **Clone this repository**

   ```bash
   git clone https://github.com/your-username/deepfake-detection.git
   cd deepfake-detection-main
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**

   ```bash
   python app.py
   ```

4. **Open your browser**

   ```
   http://127.0.0.1:5000/
   ```

---

### Model Details

* **Architecture**: EfficientNet-B1 (pretrained on ImageNet)
* **Input Shape**: 224 × 224 × 3
* **Output Classes**: Real / Fake
* **Loss Function**: Binary Cross-Entropy
* **Optimizer**: Adam
* **Metrics**: Accuracy, Precision, Recall, F1-score

The model leverages **transfer learning** from EfficientNet-B1 to achieve robust performance on relatively small datasets by fine-tuning the top layers.

---

### API Usage

After running `api.py`, you can test predictions via command line:

```bash
curl -X POST -F "file=@path_to_image.jpg" http://127.0.0.1:5000/predict
```

**Response Example:**

```json
{
  "prediction": "FAKE",
  "confidence": 0.94
}
```

---

### Results

* Achieved strong classification performance with EfficientNet-B1.
* Model effectively distinguishes subtle visual artifacts in Deepfake faces.
* High accuracy observed on both training and validation sets.

| Metric    | Value |
| --------- | ----- |
| Accuracy  | ~95%  |
| Precision | ~94%  |
| Recall    | ~96%  |
| F1-Score  | ~95%  |

---

### Web Interface

The Flask web interface allows users to:

1. Upload an image of a face.
2. Instantly get a prediction result showing whether it’s **Real** ✅ or **Fake** ❌.

---

### Requirements

See `requirements.txt` for full dependency list.
Common libraries:

```
tensorflow
flask
numpy
opencv-python
```

---

### Author

**Datta Teja**
B.Tech Student | Data Scientist & Machine Learning Engineer

---
