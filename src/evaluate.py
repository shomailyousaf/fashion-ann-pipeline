import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    processed_dir = "data/processed"
    models_dir = "models"
    x_test = np.load(os.path.join(processed_dir, "x_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    model = tf.keras.models.load_model(
        os.path.join(models_dir, "model.h5")
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        os.path.join(models_dir, "confusion_matrix.png"),
        dpi=300
    )
    plt.close()
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    }

    with open(
        os.path.join(models_dir, "metrics.json"),
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("Confusion matrix saved to models/confusion_matrix.png")
    print("Metrics saved to models/metrics.json")


if __name__ == "__main__":
    main()