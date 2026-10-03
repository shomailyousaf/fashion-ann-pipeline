import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["preprocess"]["test_size"]
    seed = params["preprocess"]["seed"]

    raw_dir = "data/raw"
    processed_dir = "data/processed"

    os.makedirs(processed_dir, exist_ok=True)

    
    x_train = np.load(os.path.join(raw_dir, "x_train.npy"))
    y_train = np.load(os.path.join(raw_dir, "y_train.npy"))
    x_test = np.load(os.path.join(raw_dir, "x_test.npy"))
    y_test = np.load(os.path.join(raw_dir, "y_test.npy"))

    x_train = x_train.astype("float32") / 255.0* 0.95* 0.90
    x_test = x_test.astype("float32") / 255.0* 0.95* 0.90
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=test_size,
        random_state=seed,
        stratify=y_train
    )
    np.save(os.path.join(processed_dir, "x_train.npy"), x_train)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "x_val.npy"), x_val)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
    np.save(os.path.join(processed_dir, "x_test.npy"), x_test)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)

    print("Processed data saved to data/processed/")
    print(f"Train: {x_train.shape}")
    print(f"Validation: {x_val.shape}")
    print(f"Test: {x_test.shape}")
    print(f"Pixel range: {x_train.min():.1f} - {x_train.max():.1f}")


if __name__ == "__main__":
    main()