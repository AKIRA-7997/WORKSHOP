# CNN Image Classifier — Fashion-MNIST

A CNN image classifier built for an AI/ML workshop using TensorFlow and Keras.

## Results

- Dataset: Fashion-MNIST (70,000 grayscale clothing images; 10 classes)
- Test accuracy: **91.90%**
- Test loss: **0.2223**

## Features

- Data preprocessing and sample visualization
- Convolutional Neural Network with batch normalization and dropout
- Training/validation accuracy and loss graphs
- Classification report and confusion matrix
- Predictions on unseen images
- Saved Keras model export

## Run

```bash
pip install tensorflow matplotlib seaborn scikit-learn numpy
python cnn_image_classifier.py
```

The trained model is saved as `fashion_mnist_cnn_classifier.keras` after training.
