from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Connect to MongoDB
client = MongoClient("mongodb+srv://mongodbconnect43:mongoDB%40123@mongodemo.wndvpbd.mongodb.net/")
db = client["FoodMate"]
users = db["Users"]
posts = db["posts"]

# 2. Prepare restaurant-level aggregated data
restaurant_data = {}

for post in posts.find():
    restaurant = post.get('restaurant')
    if not restaurant or 'name' not in restaurant:
        continue

    rname = restaurant['name']

    if rname not in restaurant_data:
        restaurant_data[rname] = {
            'total_likes': 0,
            'total_comments': 0,
            'sentiment_scores': [],
            'num_posts': 0
        }

    restaurant_data[rname]['num_posts'] += 1
    restaurant_data[rname]['total_likes'] += len(post.get('likes', []))
    restaurant_data[rname]['total_comments'] += len(post.get('comments', []))

    for comment in post.get('comments', []):
        sentiment = comment.get('sentiment')
        if sentiment == 'Positive':
            restaurant_data[rname]['sentiment_scores'].append(5.0)
        elif sentiment == 'Negative':
            restaurant_data[rname]['sentiment_scores'].append(1.0)
        elif sentiment == 'Neutral':
            restaurant_data[rname]['sentiment_scores'].append(3.0)
        # if sentiment is missing, skip

# 3. Prepare DataFrame
data = []
for rname, stats in restaurant_data.items():
    sentiments = stats['sentiment_scores']
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 3.0

    # Synthesized quality score (used for supervised training)
    score = (
        0.002 * stats['total_likes'] +
        0.01 * stats['total_comments'] +
        0.5 * avg_sentiment +
        0.01 * stats['num_posts']
    )
    score = round(min(5.0, max(1.0, score)), 2)  # Clamp between 1–5

    data.append({
        'restaurant_name': rname,
        'total_likes': stats['total_likes'],
        'total_comments': stats['total_comments'],
        'avg_sentiment': avg_sentiment,
        'num_posts': stats['num_posts'],
        'restaurant_quality_score': score
    })

df = pd.DataFrame(data)

# 4. Train Random Forest model
X = df[['total_likes', 'total_comments', 'avg_sentiment', 'num_posts']]
y = df['restaurant_quality_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 6. Save model
joblib.dump(model, 'restaurant_quality_model.pkl')
print("✅ Model saved as restaurant_quality_model.pkl")
