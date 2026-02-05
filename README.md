# 📧 Spam Email Classifier

A clean, end-to-end **Machine Learning project** for detecting spam emails using **Natural Language Processing (NLP)** techniques. This repository is designed to be **easy to run, easy to understand, and easy to extend**, making it suitable for learning, showcasing, and real-world use.

---

## 🚀 Features

* Text preprocessing pipeline (cleaning, normalization)
* Feature extraction using **Bag of Words (BoW)** 
* Supervised ML models for spam classification
* Train–test evaluation with standard metrics
* Command-line prediction support
* Clean and professional project structure

---

## 🧠 Tech Stack

* **Python 3.9+**
* **Pandas**
* **Scikit-learn**
* **Joblib**
* **Jupyter Notebook** (for experiments)

---

## 📂 Project Structure

```
spam-email-classifier/
│
├─ src/                      # Core source code
│   ├─ __init__.py
│   ├─ data_preprocessing.py # Data loading & text cleaning
│   ├─ feature_extraction.py # BoW 
│   ├─ model.py              # Model training & persistence
│   └─ predict.py            # Prediction logic
│
├─ models/                   # Saved models & vectorizers
│   ├─ spam_model.pkl
│   └─ vectorizer.pkl
│
├─ notebooks/                # Experiments & exploration
│   └─ exploration.ipynb
│
├─ train.py                  # Train the model (ENTRY POINT)
├─ predict.py                # Predict new email (ENTRY POINT)
├─ requirements.txt          # Dependencies
└─ README.md                 # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Govindakandel/spam-email-classifier.git
cd spam-email-classifier
```

---

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Run the training pipeline using:

```bash
python train.py
```

This will:

* Load and preprocess the dataset
* Extract text features
* Train the ML model
* Save the trained model and vectorizer to the `models/` directory

---

## 🔮 Predict a New Email

Use the CLI to classify an email:

```bash
python predict.py "Congratulations! You have won a free prize"
```

Example output:

```
Prediction: spam
```

---


## 🧪 Notebooks

Exploratory analysis and experiments are available in:

```
notebooks/exploration.ipynb
```



---

## 📌 Future Improvements

* Replace BoW with **TF-IDF / Word Embeddings**
* Try advanced models (SVM, Logistic Regression, Transformers)
* Add FastAPI / Flask API for deployment
* Dockerize the application
* Add CI/CD and automated tests

---

## 👤 Author

**Govinda Kandel**
Cybersecurity & AI/ML Enthusiast
GitHub: [https://github.com/Govindakandel](https://github.com/Govindakandel)

---

## ⭐ Acknowledgements

This project was built as a learning-focused .

If you find this useful, feel free to ⭐ the repository!
