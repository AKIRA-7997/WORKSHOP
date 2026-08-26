# Enhanced CNN Image Classifier — Fashion-MNIST

A complete TensorFlow/Keras image-classification project for the AI/ML workshop. It trains a convolutional neural network to recognise 10 types of clothing from the Fashion-MNIST dataset.

## Baseline

The original workshop version reached:

- Test accuracy: **91.90%**
- Test loss: **0.2223**

That result is kept as the verified baseline. The enhanced pipeline must be retrained before a new result is claimed.

## What the enhanced version adds

- Explicit **stratified train/validation split**
- Light image augmentation: translation, rotation, and zoom
- Deeper regularised CNN with batch normalization, spatial dropout, and global average pooling
- Accuracy and **top-3 accuracy** tracking
- Best-model checkpoint, early stopping, adaptive learning rate, CSV training log, and NaN protection
- Faster `tf.data` input pipeline with batching, caching, and prefetching
- Saved dataset samples, training curves, normalized confusion matrix, and prediction examples
- Automatic **highest-confidence mistake analysis**
- Saved classification report, evaluation metrics, history, run configuration, and model summary
- Reproducible seeds and deterministic TensorFlow operations when supported
- Headless-safe plotting: figures are saved without blocking a terminal run
- CLI flags for epochs, batch size, output folder, validation size, and quick smoke tests
- Single-image inference with top-3 predictions and a low-confidence warning

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Google Colab already includes most dependencies. Upload the script or clone the repository, then run it from a notebook cell with `!python cnn_image_classifier.py`.

## Train

```bash
python cnn_image_classifier.py
```

Useful options:

```bash
# Fast end-to-end smoke test
python cnn_image_classifier.py --quick-test

# Custom run
python cnn_image_classifier.py --epochs 25 --batch-size 64

# Save outputs somewhere else and also display plots
python cnn_image_classifier.py --output-dir results --show-plots
```

The first run downloads Fashion-MNIST automatically. Full training uses 20 epochs at most, but early stopping may finish sooner.

## Predict one image

After training:

```bash
python cnn_image_classifier.py \
  --predict-image path/to/clothing.png \
  --model-path artifacts/best_model.keras
```

The program prints the top three predicted classes and their confidence. External photos work best when they resemble Fashion-MNIST: one centered clothing item, grayscale, with a simple background. Bright backgrounds are inverted automatically; add `--no-auto-invert` to disable that heuristic.

## Generated outputs

Training creates an `artifacts/` folder containing:

```text
artifacts/
├── best_model.keras
├── final_model.keras
├── classification_report.json
├── classification_report.txt
├── confusion_matrix.png
├── dataset_samples.png
├── evaluation_metrics.json
├── model_summary.txt
├── run_config.json
├── sample_predictions.png
├── strongest_mistakes.png
├── training_curves.png
├── training_history.json
└── training_log.csv
```

Large generated model files and run artifacts are intentionally ignored by Git. Commit only selected results if they are needed for a demo.

## Classes

`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, and `Ankle boot`.

## Notes

- Fashion-MNIST is balanced, but the explicit stratified split makes validation intent clear and reusable.
- Data augmentation and regularization are intended to improve generalization; they do not guarantee a specific score on every machine.
- The most commonly confused categories are usually `Shirt`, `T-shirt/top`, `Pullover`, and `Coat`. The generated mistake plot helps diagnose them.
