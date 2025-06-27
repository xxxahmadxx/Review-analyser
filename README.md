# Review-analyser

A complete end-to-end sentiment analysis pipeline built in Python, using a Bag-of-Words model and a fully-connected neural network (TensorFlow/Keras).  
Trained and tested on real Amazon reviews, this project can predict whether any review is **positive** or **negative**—with live GPU acceleration (RTX 3080 supported!).

---

## Features

- **Bag-of-Words preprocessing** (with 1,500-word vocabulary)
- **Simple, fast neural network** (customizable hidden layers)
- **Efficient handling of large datasets** (with scikit-learn + NumPy)
- **Predicts sentiment for any new review** (demo included)
- **Runs on CPU or NVIDIA GPU (with CUDA/cuDNN support)**

---

## Model Architecture

- **Input:** Bag-of-Words vector (size 1,500)
- **Hidden layer:** 16 ReLU neurons, with optional Dropout
- **Output layer:** 1 neuron, sigmoid activation
- **Loss:** Binary cross-entropy
- **Optimizer:** Adam

---

## Setup & Usage

### 1. Install requirements
- Install TensorFlow: see [TensorFlow install instructions](https://www.tensorflow.org/install/pip)
- Other dependencies:
    ```bash
    pip install numpy pandas scikit-learn
    ```

### 2. Preprocess the data *(if needed)*
- Place your `labeled_reviews.csv` in the `data/` directory
- Or, create it with:
    ```bash
    python scripts/preprocess.py
    ```

### 3. Train the model
```bash
python scripts/train_model.py
```

### 4. Predict using the trained model
```bash
python scripts/use.py
```

