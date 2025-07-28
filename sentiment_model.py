import pandas as pd
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load IMDB sentiment dataset (pos/neg reviews)
from sklearn.datasets import load_files
import os
import urllib.request
import tarfile

# Download the IMDB dataset (first time only)
def download_imdb_dataset():
    if not os.path.exists("aclImdb"):
        print("Downloading IMDB dataset...")
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        urllib.request.urlretrieve(url, "aclImdb_v1.tar.gz")
        print("Extracting dataset...")
        with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
            tar.extractall()

download_imdb_dataset()

# Load only train data (unlabeled test set is ignored here)
reviews = load_files("aclImdb/train", categories=["pos", "neg"], encoding="utf-8")
X, y = reviews.data, reviews.target  # y: 1 = pos, 0 = neg

# Convert labels to stars (1 to 5) to simulate HuggingFace model
def convert_to_stars(label):  # 0 = negative, 1 = positive
    return 1 if label == 0 else 5

stars = [convert_to_stars(label) for label in y]

# Convert stars to sentiment labels
def stars_to_sentiment(stars):
    if stars <= 2:
        return "Negative"
    elif stars == 3:
        return "Neutral"
    else:
        return "Positive"

sentiments = [stars_to_sentiment(s) for s in stars]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)

# 3. Train TF-IDF + Logistic Regression model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)

# 4. Evaluate
print("📊 Classification Report:")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 5. Predict Sentiment Function (just like your HuggingFace version)
def get_sentiment_label(text):
    prediction = pipeline.predict([text])[0]
    return prediction

# 6. Try Examples
print(get_sentiment_label("This food was amazing and full of flavor!"))  # Positive
print(get_sentiment_label("Just okay, nothing special."))                # Neutral (if added)
print(get_sentiment_label("I hated the dish, worst experience ever."))   # Negative