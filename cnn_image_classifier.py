"""Fashion-MNIST CNN image classifier for the AI/ML workshop."""

import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def show_samples(images, labels, class_names):
    plt.figure(figsize=(10, 6))
    for index in range(15):
        plt.subplot(3, 5, index + 1)
        plt.imshow(images[index].squeeze(), cmap="gray")
        plt.title(class_names[labels[index]])
        plt.axis("off")
    plt.suptitle("Sample Fashion-MNIST Images", fontsize=16)
    plt.tight_layout()
    plt.show()


def build_model():
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.30),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.40),
            layers.Dense(10, activation="softmax"),
        ],
        name="fashion_mnist_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], marker="o", label="Training Loss")
    plt.plot(history.history["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_confusion_matrix(actual, predicted, class_names):
    matrix = confusion_matrix(actual, predicted)
    plt.figure(figsize=(11, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def show_predictions(images, actual_labels, probabilities, class_names):
    sample_indices = np.random.choice(len(images), 15, replace=False)
    predicted_labels = np.argmax(probabilities, axis=1)
    plt.figure(figsize=(12, 8))

    for position, image_index in enumerate(sample_indices):
        plt.subplot(3, 5, position + 1)
        plt.imshow(images[image_index].squeeze(), cmap="gray")
        actual = class_names[actual_labels[image_index]]
        predicted = class_names[predicted_labels[image_index]]
        confidence = probabilities[image_index].max() * 100
        colour = "green" if actual == predicted else "red"
        plt.title(
            f"Actual: {actual}\nPred: {predicted}\n{confidence:.1f}% confidence",
            color=colour,
            fontsize=8,
        )
        plt.axis("off")

    plt.suptitle("CNN Predictions on Unseen Test Images", fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices("GPU"))

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]

    x_train = np.expand_dims(x_train.astype("float32") / 255.0, axis=-1)
    x_test = np.expand_dims(x_test.astype("float32") / 255.0, axis=-1)
    print("Training images:", x_train.shape)
    print("Testing images :", x_test.shape)
    show_samples(x_train, y_train, class_names)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=8,
        batch_size=128,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss : {test_loss:.4f}")
    print(f"Test accuracy : {test_accuracy * 100:.2f}%")
    plot_history(history)

    probabilities = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    print("\nClassification Report:\n")
    print(classification_report(y_test, predicted_labels, target_names=class_names))
    show_confusion_matrix(y_test, predicted_labels, class_names)
    show_predictions(x_test, y_test, probabilities, class_names)

    model.save("fashion_mnist_cnn_classifier.keras")
    print("\nSaved model: fashion_mnist_cnn_classifier.keras")


if __name__ == "__main__":
    main()
