Here’s a clean and professional **README.md** tailored for your project setup 👇

---

# Breast Cancer Classification – Random Forest Model

## Project Overview

This project trains a **Random Forest Classifier** to predict whether a tumor is **malignant (M)** or **benign (B)** using the Breast Cancer dataset.
The workflow covers:

* Data preprocessing and encoding
* Train/test split
* Model training and evaluation
* Model serialization with `joblib`
* Docker containerization for reproducibility

---

## Folder Structure

```
Lab1/
│
├── dockerfile
├── ReadMe.md
└── src/
    ├── main.py
    ├── breast_cancer_data.csv
    └── requirements.txt
```

---

## Setup Instructions

### **Install Dependencies Locally**

If running without Docker:

```bash
cd src
pip install -r requirements.txt
python main.py
```

---

### ** Build and Run with Docker**

#### Build the Image

From the `Lab1` directory:

```bash
docker build -t breast-cancer-train .
```

#### Run the Container

```bash
docker run --rm breast-cancer-train
```

#### (Optional) Save Model Outside the Container

To persist the trained `.pkl` model on your host:

```bash
docker run --rm -v "%cd%/model:/app/model" breast-cancer-train
```

The model will be saved to:

```
Lab1/model/breast_cancer_model.pkl
```

---

## Requirements

The required Python libraries are listed in `requirements.txt`:

```
pandas
scikit-learn
joblib
```

You can install them using:

```bash
pip install -r requirements.txt
```

---

##  Script Workflow (`main.py`)

1. **Load Dataset:** Reads `breast_cancer_data.csv`.
2. **Clean Data:** Removes unnecessary columns (`id`, unnamed), encodes `diagnosis` (M → 1, B → 0).
3. **Preprocess:** Handles categorical and missing values.
4. **Train/Test Split:** 80% training, 20% testing.
5. **Model Training:** Random Forest with 100 trees.
6. **Evaluation:** Prints test accuracy.
7. **Save Model:** Saves trained model as `breast_cancer_model.pkl`.

---

## Example Output

```
Data sample:
[5 rows x 33 columns]
✅ Training complete. Test Accuracy: 0.974
💾 Model saved to: model/breast_cancer_model.pkl
```

---

