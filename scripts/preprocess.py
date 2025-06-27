import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re

#load data
df = pd.read_csv('data/labeled_reviews.csv')
texts = df['review'].astype(str).tolist()
labels = (df['sentiment'] == 'Positive').astype(int).values

#build vocab
def build_vocab(texts, vocab_size=1500):
    word_freq = defaultdict(int)
    for text in texts:
        for word in re.findall(r'\b\w+\b', text.lower()):
            word_freq[word] += 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    return {word: idx for idx, (word, _) in enumerate(sorted_words[:vocab_size])}

word_index = build_vocab(texts, vocab_size=1500)

#convert textx to bag of words
def texts_to_bow(texts, word_index):
    vectors = np.zeros((len(texts), len(word_index)), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r'\b\w+\b', text.lower()):
            if word in word_index:
                vectors[i, word_index[word]] += 1.0
    
    return vectors

X = texts_to_bow(texts, word_index)
y = labels

#split into train and eval
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
