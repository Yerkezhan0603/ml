# Toxic Comment Classifier (Logistic Regression)

This project is a full-stack machine learning application that classifies user comments as toxic or non-toxic across multiple categories. It uses **TF-IDF vectorization** and **Logistic Regression** in a **One-vs-Rest** setup. The model is trained on the Jigsaw Toxic Comment dataset.

---

## 📦 Features

- Multi-label classification (predicts 1 or more of 6 labels)
- TF-IDF + Logistic Regression model
- FastAPI backend + HTML frontend
- Real-time prediction via web UI

---

## 🏗️ Project Structure
```
toxic_classifier/
├── backend/
│   ├── main.py              # FastAPI backend logic
│   ├── model.joblib         # Trained logistic regression model
│   ├── tfidf.joblib         # TF-IDF vectorizer
├── data/
│   ├── train_clean.csv      # Preprocessed training dataset
│   ├── test_clean.csv       # Cleaned test comments
│   ├── test_labels.csv      # Ground truth test labels
├── frontend/
│   ├── index.html           # HTML interface (text input + results)
├── train_model.py           # Training script
├── requirements.txt         # Python dependencies
├── README.md
```

---

## 🚀 Running the App Locally

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (optional)
```bash
python train_model.py
```

### 4. Start the app
```bash
uvicorn backend.main:app --reload
```

Access the app at: [http://localhost:8000](http://localhost:8000)

---

## 🧪 Example Input
```
Input: You're an absolute disgrace and a joke.
Output: ['toxic', 'insult']
```

---

## ⚙️ Model Details
- Text vectorization: TF-IDF (top 10,000 features)
- Classifier: Logistic Regression using OneVsRestClassifier
- Multi-label outputs: 6 toxic labels
- Threshold: fixed (e.g., 0.3)
