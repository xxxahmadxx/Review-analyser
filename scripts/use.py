from tensorflow import keras
import numpy as np
import re
import preprocess

model = keras.models.load_model('models/sentiment_model.keras')

def preprocess_text(text, word_index):
    vec = np.zeros((1, len(word_index)), dtype=np.float32)
    for word in re.findall(r'\b\w+\b', text.lower()):
        if word in word_index:
            vec[0, word_index[word]] += 1.0
    return vec

def predict_sentiment(text, model, word_index):
    bow_vec = preprocess_text(text, word_index)
    prob = model.predict(bow_vec)[0, 0]
    sentiment = 'Positive' if prob > 0.5 else 'Negative'
    print(f"Review: {text}\nPredicted sentiment: {sentiment} (probability: {prob:.3f})")
    return sentiment, prob

print("Enter you review: ")
review = input()
predict_sentiment(review, model, preprocess.word_index)