import tensorflow as tf
from tensorflow import keras
import preprocess
import matplotlib.pyplot as plt


#build mmodel
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(1500,)),
    keras.layers.Dropout(0.6),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    preprocess.X_train, preprocess.y_train,
    epochs=6,
    batch_size=64,
    validation_data=(preprocess.X_val, preprocess.y_val)
)

loss, acc = model.evaluate(preprocess.X_val, preprocess.y_val)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}")

model.save('sentiment_model.keras')