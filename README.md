# MNIST Digit Classification — Perceptron vs ANN vs CNN

A comparative study of three neural network architectures — **Perceptron**, **Artificial Neural Network (ANN)**, and **Convolutional Neural Network (CNN)** — trained on the MNIST handwritten digit dataset using **TensorFlow/Keras**, to analyze which architecture performs best and why.

---

## 📌 Project Overview

The MNIST dataset (60,000 training images + 10,000 test images of handwritten digits 0–9, each 28x28 grayscale) is a classic benchmark for image classification. This project builds and trains three models of increasing complexity on the same dataset to:

- Understand how model architecture affects performance on image data
- Compare a single-layer Perceptron, a fully-connected ANN, and a CNN
- Visualize and quantify why CNNs are generally better suited for image classification tasks
- Serve as a hands-on demonstration of core deep learning concepts (dense layers, activation functions, convolution, pooling)

---

## 🧠 Models Implemented

### 1. Perceptron (Single-Layer)
- A single dense layer with no hidden layers, acting as the simplest linear classifier.
- Flattens the 28x28 image into a 784-length vector.
- Uses a `softmax` output layer for 10-class classification.
- Serves as the baseline to show the limitations of linear models on non-linear image data.

### 2. Artificial Neural Network (ANN)
- A fully-connected feedforward network with one or more hidden layers.
- Uses `ReLU` activations in hidden layers and `softmax` in the output layer.
- Captures non-linear relationships that the Perceptron cannot, but still treats pixels independently (no spatial awareness).

### 3. Convolutional Neural Network (CNN)
- Uses `Conv2D` + `MaxPooling2D` layers to extract spatial features (edges, curves, strokes).
- Followed by `Flatten` and `Dense` layers for final classification.
- Preserves spatial structure of the image, making it the most effective architecture for this task.

---

## 🛠️ Tech Stack

| Component      | Tool/Library         |
|-----------------|----------------------|
| Environment      | Google Colab (GPU runtime) |
| Language         | Python               |
| Deep Learning    | TensorFlow / Keras   |
| Data Handling    | NumPy                |
| Visualization    | Matplotlib / Seaborn |
| Dataset          | MNIST (via `tensorflow.keras.datasets`) |

---

## 📂 Project Structure

This project is built and run entirely on **Google Colab** as a single notebook:

```
MNIST[Perceptron Vs ANN Vs CNN].ipynb   # Full notebook: data loading, all 3 models, training, comparison, visualizations
```

## ⚙️ Setup & Installation (Google Colab)

No local setup required — everything runs in the browser via Colab's free GPU/TPU runtime.

1. Open the notebook: **[Open in Google Colab](https://colab.research.google.com/drive/1bSWfpKjyqJ5sfGbSAnVT1gM-KGEX_bTd?usp=sharing)>)**
2. (Recommended) Switch to a GPU runtime for faster training:
   `Runtime → Change runtime type → Hardware accelerator → GPU (T4)`
3. Run all cells top to bottom:
   `Runtime → Run all`

TensorFlow, NumPy, Matplotlib, and Seaborn come pre-installed on Colab, so no `pip install` is needed unless you add extra libraries. If you do, just include a setup cell at the top:

```python
!pip install -q scikit-learn seaborn
```

## ▶️ Usage

The notebook is organized into clearly labeled sections/cells:

1. **Data Loading & Preprocessing** — normalizing pixel values, reshaping for each model
2. **Perceptron Model** — build, compile, train, evaluate
3. **ANN Model** — build, compile, train, evaluate
4. **CNN Model** — build, compile, train, evaluate
5. **Comparison & Visualization** — accuracy/loss plots and confusion matrices across all three models

Just run the notebook cells in order — each model trains and evaluates independently, and the final section compares all three side by side.

To save your trained models from Colab (optional):
```python
model.save('cnn_model.h5')
from google.colab import files
files.download('cnn_model.h5')
```

---

## 📊 Results

| Model       | Test Accuracy | Test Loss | 
|-------------|:--------------:|:---------:|
| Perceptron   | 90.90%        | 2.45   | 
| ANN            | 97.97%        | 6.85   | 
| CNN            | 99.24%        | 33.61   |



### Key Observations
- The **Perceptron** struggles to capture non-linear patterns in pixel data, resulting in the lowest accuracy among the three.
- The **ANN** improves performance significantly by introducing hidden layers and non-linear activations, but still ignores spatial relationships between pixels.
- The **CNN** achieves the highest accuracy by using convolutional filters to detect local spatial patterns (edges, curves), making it the best-suited architecture for image data.


## 📈 Visualizations

- **Accuracy comparison bar chart** across the three models
- **Loss curves** (training vs validation) for each model

---

## 🔮 Future Improvements

- Add data augmentation to improve CNN generalization
- Experiment with deeper CNN architectures (e.g., adding more Conv2D layers, Dropout, BatchNorm)
- Try transfer learning with pretrained models for comparison
- Deploy the best-performing model via a Streamlit app for live digit recognition

---

## 🙋 Author

**Hardik Sikka**
B.Tech CSE (AI/ML), VIT Bhopal — 2nd Year
Building projects like this as part of my journey toward a career in AI/Data Science.
[GitHub](https://github.com/hardik020106) | [LinkedIn](https://linkedin.com/in/hardiksikka)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
