from transformers import pipeline

# Load the model once
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def get_sentiment_label(text):
    result = sentiment_pipeline(text[:512])[0]  # truncate long texts
    label = result['label']  # e.g. "3 stars"
    stars = int(label[0])
    if stars <= 2:
        return "Negative"
    elif stars == 3:
        return "Neutral"
    else:
        return "Positive"
