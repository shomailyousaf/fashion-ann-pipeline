import os
import numpy as np
import yaml
import tensorflow as tf


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    dense_units = params["train"]["dense_units"]
    dropout_rate = params["train"]["dropout_rate"]
    learning_rate = params["train"]["learning_rate"]
    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]

    processed_dir = "data/processed"
    models_dir = "models"

    os.makedirs(models_dir, exist_ok=True)
    x_train = np.load(os.path.join(processed_dir, "x_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    x_val = np.load(os.path.join(processed_dir, "x_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    model.save(os.path.join(models_dir, "model.h5"))

    
    history_array = np.column_stack([
        history.history["loss"],
        history.history["accuracy"],
        history.history["val_loss"],
        history.history["val_accuracy"]
    ])

    np.savetxt(
        os.path.join(models_dir, "history.csv"),
        history_array,
        delimiter=",",
        header="loss,accuracy,val_loss,val_accuracy",
        comments=""
    )

    print("Model saved to models/model.h5")
    print("Training history saved to models/history.csv")


if __name__ == "__main__":
    main()