from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from joblib import dump

# 1. MongoDB connection
client = MongoClient("mongodb+srv://mongodbconnect43:mongoDB%40123@mongodemo.wndvpbd.mongodb.net/")
db = client["FoodMate"]
collection = db["sample_data_comment_sentiment"]

# 2. Load labeled comment data from MongoDB
cursor = collection.find({
"comment": {"$exists": True, "$ne": ""},
"label": {"$in": [0, 1, 2]} # 0 = Negative, 1 = Positive, 2 = Neutral
})
data = list(cursor)

if not data:
   raise ValueError("No labeled comments found in MongoDB.")

# 3. Convert to DataFrame
df = pd.DataFrame(data)
df = df[["comment", "label"]].dropna()

# 4. Map back for printing
label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
inv_map = {v: k for k, v in label_map.items()}

# 5. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
df["comment"], df["label"], test_size=0.2, random_state=42
)

# 6. TF-IDF + Naive Bayes pipeline
pipeline = Pipeline([
("tfidf", TfidfVectorizer(stop_words="english")),
("nb", MultinomialNB())
])

# 7. Train model
pipeline.fit(X_train, y_train)

# 8. Evaluate model
y_pred = pipeline.predict(X_test)
print("Model Evaluation:")
print(classification_report(y_test, y_pred, target_names=label_map.values()))

# 9. Save trained model
dump(pipeline, "sentiment_model.joblib")

# 10. Inference function
def predict_sentiment(text):
    pred_label = pipeline.predict([text])[0]
    return label_map[pred_label]

# 11. Optional test
if __name__ == "__main__":
    test_comments = [
        "The food was really bad!",
        "Too expensive for the quality.",
        "It was okay, nothing special.",
        "Absolutely loved the flavors and presentation!"
    ]
    for comment in test_comments:
        print(f"{comment} → {predict_sentiment(comment)}")