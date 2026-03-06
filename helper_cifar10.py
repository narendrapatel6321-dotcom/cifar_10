import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_training_curve(csv_path):
    """
    Plot training and validation loss/accuracy curves from a Keras CSVLogger file.

    Generates a two-panel figure:
      - Top panel:    Train vs. validation loss, with a marker at the best (lowest) val_loss epoch.
      - Bottom panel: Train vs. validation accuracy.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file produced by Keras's CSVLogger callback.
        Expected columns: 'epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy'.

    Returns
    -------
    None
        Displays the plot inline.

    Example
    -------
    >>> plot_training_curve("logs/training.csv")
    """
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)
    epochs = df.index

    plt.style.use("seaborn-v0_8")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ── Loss panel ────────────────────────────────────────────
    ax1.plot(epochs, df["loss"],     linestyle="-",  linewidth=2, label="train_loss")
    ax1.plot(epochs, df["val_loss"], linestyle="--", linewidth=2, label="val_loss")

    best_epoch = df["val_loss"].idxmin()
    ax1.scatter(best_epoch, df["val_loss"].iloc[best_epoch],
                s=80, zorder=5, label=f"best val_loss (epoch {best_epoch})")
    ax1.axvline(best_epoch, linestyle=":", alpha=0.6)

    loss_min = min(df["loss"].min(), df["val_loss"].min())
    loss_max = max(df["loss"].max(), df["val_loss"].max())
    margin = 0.05 * (loss_max - loss_min)
    ax1.set_ylim(loss_min - margin, loss_max + margin)
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # ── Accuracy panel ────────────────────────────────────────
    ax2.plot(epochs, df["accuracy"],     linestyle="-",  linewidth=2, label="train_accuracy")
    ax2.plot(epochs, df["val_accuracy"], linestyle="--", linewidth=2, label="val_accuracy")

    acc_min = min(df["accuracy"].min(), df["val_accuracy"].min())
    acc_max = max(df["accuracy"].max(), df["val_accuracy"].max())
    margin = 0.05 * (acc_max - acc_min)
    ax2.set_ylim(acc_min - margin, acc_max + margin)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_model(model, x_test, y_test, class_names):
    """
    Evaluate a multi-class image classification model and display results.

    Prints test loss and accuracy, a per-class classification report,
    and renders a confusion matrix heatmap.

    Designed for CIFAR-10 (10-class softmax output). Predictions are
    determined via argmax (not sigmoid threshold).

    Parameters
    ----------
    model : tf.keras.Model
        A trained Keras model with softmax output (10 units).
    x_test : np.ndarray
        Test images, shape (N, 32, 32, 3), already normalised to [0, 1].
    y_test : np.ndarray
        Integer class labels, shape (N,) or (N, 1).
    class_names : list of str
        Display names for all 10 classes, ordered by label index.
        Example: ['airplane', 'automobile', ..., 'truck']

    Returns
    -------
    None
        Prints metrics and displays the confusion matrix inline.

    Example
    -------
    >>> evaluate_model(model, x_test, y_test, class_names)
    """
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)   # <-- argmax for multi-class
    y_true = y_test.flatten()                     # ensure shape (N,)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
