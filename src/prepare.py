import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


def main():
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Fashion-MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    np.save(os.path.join(output_dir, "x_train.npy"), x_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "x_test.npy"), x_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    print("Fashion-MNIST raw data saved to data/raw/")
    print(f"Training images: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}")
    print(f"Test labels: {y_test.shape}")


if __name__ == "__main__":
    main()