# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, hamming_loss
import joblib

# Load dataset
df = pd.read_csv('data/train_clean.csv')

# Fill NA
df['comment_text'] = df['comment_text'].fillna("")

# Define features and labels
X = df['comment_text']
y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("F1 score (macro):", f1_score(y_test, y_pred, average='macro'))
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=y.columns))

# Save model and vectorizer
joblib.dump(model, 'backend/model.joblib')
joblib.dump(vectorizer, 'backend/tfidf.joblib')
