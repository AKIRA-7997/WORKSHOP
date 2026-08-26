"""Enhanced Fashion-MNIST CNN classifier for the AI/ML workshop.

Train a regularized CNN, save evaluation artifacts, inspect its strongest
mistakes, or use a saved model to classify one external clothing image.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY") and "google.colab" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers


CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or run inference with an enhanced Fashion-MNIST CNN."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Folder for models, metrics, logs, and plots.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots in addition to saving them.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use a small stratified subset and two epochs for a fast smoke test.",
    )
    parser.add_argument(
        "--predict-image",
        type=Path,
        help="Skip training and classify one external image with a saved model.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/best_model.keras"),
        help="Saved Keras model used by --predict-image.",
    )
    parser.add_argument(
        "--no-auto-invert",
        action="store_true",
        help="Do not auto-invert bright-background external images.",
    )
    return parser.parse_args()


def set_global_determinism(seed: int) -> None:
    """Seed common random sources and request deterministic TensorFlow ops."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def normalise_images(images: np.ndarray) -> np.ndarray:
    return np.expand_dims(images.astype("float32") / 255.0, axis=-1)


def load_datasets(
    validation_size: float,
    seed: int,
    quick_test: bool,
) -> tuple[np.ndarray, ...]:
    """Load Fashion-MNIST and create an explicit stratified validation split."""
    if not 0.05 <= validation_size <= 0.40:
        raise ValueError("--validation-size must be between 0.05 and 0.40")

    try:
        (x_full, y_full), (x_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )
    except Exception as error:
        raise RuntimeError(
            "Fashion-MNIST could not be downloaded. Check the internet connection "
            "and try again."
        ) from error

    if quick_test:
        x_full, _, y_full, _ = train_test_split(
            x_full,
            y_full,
            train_size=6_000,
            stratify=y_full,
            random_state=seed,
        )
        x_test, _, y_test, _ = train_test_split(
            x_test,
            y_test,
            train_size=1_500,
            stratify=y_test,
            random_state=seed,
        )

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_full,
        y_full,
        test_size=validation_size,
        stratify=y_full,
        random_state=seed,
    )

    return (
        normalise_images(x_train),
        y_train,
        normalise_images(x_validation),
        y_validation,
        normalise_images(x_test),
        y_test,
    )


def make_tf_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    autotune = tf.data.AUTOTUNE
    train_data = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(autotune)
    )
    validation_data = (
        tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
        .batch(batch_size)
        .cache()
        .prefetch(autotune)
    )
    test_data = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .cache()
        .prefetch(autotune)
    )
    return train_data, validation_data, test_data


def convolution_block(
    inputs: tf.Tensor,
    filters: int,
    dropout_rate: float,
) -> tf.Tensor:
    """Two convolution layers followed by pooling and spatial dropout."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    return layers.SpatialDropout2D(dropout_rate)(x)


def build_model() -> tf.keras.Model:
    """Build a compact, regularized CNN with train-time augmentation."""
    augmentation = tf.keras.Sequential(
        [
            layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
            layers.RandomRotation(0.04, fill_mode="nearest"),
            layers.RandomZoom(0.08, 0.08, fill_mode="nearest"),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(shape=(28, 28, 1), name="image")
    x = augmentation(inputs)
    x = convolution_block(x, filters=32, dropout_rate=0.12)
    x = convolution_block(x, filters=64, dropout_rate=0.18)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_probs")(x)

    model = tf.keras.Model(inputs, outputs, name="enhanced_fashion_mnist_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=3, name="top_3_accuracy"
            ),
        ],
    )
    return model


def save_figure(path: Path, show_plots: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()


def plot_samples(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    show_plots: bool,
) -> None:
    plt.figure(figsize=(10, 6))
    for index in range(15):
        plt.subplot(3, 5, index + 1)
        plt.imshow(images[index].squeeze(), cmap="gray")
        plt.title(CLASS_NAMES[int(labels[index])])
        plt.axis("off")
    plt.suptitle("Sample Fashion-MNIST Images", fontsize=16)
    save_figure(output_path, show_plots)


def plot_history(
    history: tf.keras.callbacks.History,
    output_path: Path,
    show_plots: bool,
) -> None:
    epochs = range(1, len(history.history["loss"]) + 1)
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history.history["accuracy"], label="Train")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.history["loss"], label="Train")
    plt.plot(epochs, history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history.history["top_3_accuracy"], label="Train")
    plt.plot(epochs, history.history["val_top_3_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Top-3 accuracy")
    plt.title("Top-3 Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    save_figure(output_path, show_plots)


def plot_confusion_matrix(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    show_plots: bool,
) -> None:
    matrix = confusion_matrix(actual, predicted, labels=range(len(CLASS_NAMES)))
    row_totals = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    normalised = matrix / row_totals

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        normalised,
        annot=matrix,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Fraction of actual class"},
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Confusion Matrix (counts annotated, rows normalised)")
    plt.xticks(rotation=45, ha="right")
    save_figure(output_path, show_plots)


def plot_predictions(
    images: np.ndarray,
    actual_labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    show_plots: bool,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(images), 15, replace=False)
    predicted_labels = np.argmax(probabilities, axis=1)
    plt.figure(figsize=(12, 9))

    for position, image_index in enumerate(sample_indices):
        plt.subplot(3, 5, position + 1)
        plt.imshow(images[image_index].squeeze(), cmap="gray")
        actual = CLASS_NAMES[int(actual_labels[image_index])]
        predicted = CLASS_NAMES[int(predicted_labels[image_index])]
        confidence = float(probabilities[image_index].max() * 100)
        colour = "green" if actual == predicted else "red"
        plt.title(
            f"Actual: {actual}\nPred: {predicted}\n{confidence:.1f}%",
            color=colour,
            fontsize=8,
        )
        plt.axis("off")

    plt.suptitle("Predictions on Unseen Test Images", fontsize=16)
    save_figure(output_path, show_plots)


def plot_strongest_mistakes(
    images: np.ndarray,
    actual_labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Save the most confident wrong answers for useful error analysis."""
    predicted = np.argmax(probabilities, axis=1)
    wrong_indices = np.flatnonzero(predicted != actual_labels)
    if not len(wrong_indices):
        return

    wrong_confidence = probabilities[wrong_indices, predicted[wrong_indices]]
    strongest = wrong_indices[np.argsort(wrong_confidence)[::-1][:15]]
    plt.figure(figsize=(12, 9))

    for position, image_index in enumerate(strongest):
        plt.subplot(3, 5, position + 1)
        plt.imshow(images[image_index].squeeze(), cmap="gray")
        actual = CLASS_NAMES[int(actual_labels[image_index])]
        guess = CLASS_NAMES[int(predicted[image_index])]
        confidence = float(probabilities[image_index, predicted[image_index]] * 100)
        plt.title(
            f"Actual: {actual}\nGuess: {guess}\n{confidence:.1f}%",
            color="red",
            fontsize=8,
        )
        plt.axis("off")

    plt.suptitle("Highest-Confidence Mistakes", fontsize=16)
    save_figure(output_path, show_plots)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_model_summary(model: tf.keras.Model, output_path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_and_save(
    model: tf.keras.Model,
    test_data: tf.data.Dataset,
    x_test: np.ndarray,
    y_test: np.ndarray,
    history: tf.keras.callbacks.History,
    output_dir: Path,
    show_plots: bool,
    seed: int,
) -> dict[str, float]:
    metrics = {
        key: float(value)
        for key, value in model.evaluate(test_data, verbose=0, return_dict=True).items()
    }
    probabilities = model.predict(test_data, verbose=0)
    predicted = np.argmax(probabilities, axis=1)

    report_text = classification_report(
        y_test,
        predicted,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    report_json = classification_report(
        y_test,
        predicted,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    write_json(output_dir / "evaluation_metrics.json", metrics)
    write_json(output_dir / "classification_report.json", report_json)
    write_json(
        output_dir / "training_history.json",
        {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    )
    (output_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    plot_history(history, output_dir / "training_curves.png", show_plots)
    plot_confusion_matrix(
        y_test, predicted, output_dir / "confusion_matrix.png", show_plots
    )
    plot_predictions(
        x_test,
        y_test,
        probabilities,
        output_dir / "sample_predictions.png",
        show_plots,
        seed,
    )
    plot_strongest_mistakes(
        x_test,
        y_test,
        probabilities,
        output_dir / "strongest_mistakes.png",
        show_plots,
    )

    print("\nClassification report:\n")
    print(report_text)
    return metrics


def prepare_external_image(image_path: Path, auto_invert: bool) -> np.ndarray:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = tf.keras.utils.load_img(
        image_path,
        color_mode="grayscale",
        target_size=(28, 28),
    )
    array = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
    if auto_invert and float(array.mean()) > 0.5:
        array = 1.0 - array
    return np.expand_dims(array, axis=0)


def predict_one_image(
    model_path: Path,
    image_path: Path,
    auto_invert: bool,
) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the classifier first."
        )
    model = tf.keras.models.load_model(model_path)
    image = prepare_external_image(image_path, auto_invert=auto_invert)
    probabilities = model.predict(image, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]

    print(f"\nPrediction for: {image_path}")
    for rank, index in enumerate(top_indices, start=1):
        print(f"{rank}. {CLASS_NAMES[int(index)]}: {probabilities[index] * 100:.2f}%")
    if float(probabilities[top_indices[0]]) < 0.60:
        print("Warning: low confidence; this image may differ from Fashion-MNIST style.")


def train(args: argparse.Namespace) -> None:
    set_global_determinism(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epochs = 2 if args.quick_test else args.epochs

    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("Output directory:", args.output_dir.resolve())

    data = load_datasets(args.validation_size, args.seed, args.quick_test)
    x_train, y_train, x_validation, y_validation, x_test, y_test = data
    print("Training images  :", x_train.shape)
    print("Validation images:", x_validation.shape)
    print("Testing images   :", x_test.shape)

    plot_samples(
        x_train, y_train, args.output_dir / "dataset_samples.png", args.show_plots
    )
    train_data, validation_data, test_data = make_tf_datasets(
        *data,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = build_model()
    model.summary()
    save_model_summary(model, args.output_dir / "model_summary.txt")
    write_json(
        args.output_dir / "run_config.json",
        {
            "epochs_requested": epochs,
            "batch_size": args.batch_size,
            "validation_size": args.validation_size,
            "seed": args.seed,
            "quick_test": args.quick_test,
            "tensorflow_version": tf.__version__,
            "training_examples": len(x_train),
            "validation_examples": len(x_validation),
            "test_examples": len(x_test),
        },
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output_dir / "best_model.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(args.output_dir / "training_log.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    metrics = evaluate_and_save(
        model,
        test_data,
        x_test,
        y_test,
        history,
        args.output_dir,
        args.show_plots,
        args.seed,
    )
    model.save(args.output_dir / "final_model.keras")

    print(f"\nTest loss         : {metrics['loss']:.4f}")
    print(f"Test accuracy     : {metrics['accuracy'] * 100:.2f}%")
    print(f"Test top-3 accuracy: {metrics['top_3_accuracy'] * 100:.2f}%")
    print(f"Saved all outputs to: {args.output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive integers")

    if args.predict_image:
        set_global_determinism(args.seed)
        predict_one_image(
            args.model_path,
            args.predict_image,
            auto_invert=not args.no_auto_invert,
        )
        return
    train(args)


if __name__ == "__main__":
    main()
