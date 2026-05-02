#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a VGG16 model to classify breast cancer mammograms into BI-RADS 1-4."
    )
    parser.add_argument("--data-dir", default="data", help="Dataset root folder containing birads1-4 subfolders.")
    parser.add_argument("--output-dir", default="outputs/vgg16_birads", help="Directory for model artifacts.")
    parser.add_argument("--img-height", type=int, default=224, help="Input image height.")
    parser.add_argument("--img-width", type=int, default=224, help="Input image width.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=12, help="Initial training epochs with frozen VGG16 base.")
    parser.add_argument("--fine-tune-epochs", type=int, default=6, help="Extra fine-tuning epochs.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5, help="Fine-tuning learning rate.")
    parser.add_argument(
        "--weights",
        default="imagenet",
        choices=["imagenet", "none"],
        help="VGG16 initialization weights. Use 'none' for offline testing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets and model, then exit before training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def count_images_per_class(data_dir: Path) -> dict[str, int]:
    counts = {}
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        counts[class_dir.name] = sum(1 for file_path in class_dir.iterdir() if file_path.is_file())
    return counts


def collect_filepaths_and_labels(data_dir: Path) -> tuple[list[str], np.ndarray, list[str]]:
    class_names = sorted(class_dir.name for class_dir in data_dir.iterdir() if class_dir.is_dir())
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

    filepaths: list[str] = []
    labels: list[int] = []
    for class_name in class_names:
        class_dir = data_dir / class_name
        for file_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
            filepaths.append(str(file_path))
            labels.append(class_to_index[class_name])

    return filepaths, np.array(labels, dtype=np.int32), class_names


def load_and_preprocess_image(file_path: tf.Tensor, label: tf.Tensor, image_size: tuple[int, int]):
    image_bytes = tf.io.read_file(file_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(
    filepaths: list[str],
    labels: np.ndarray,
    image_size: tuple[int, int],
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda path, label: load_and_preprocess_image(path, label, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_datasets(args: argparse.Namespace):
    image_size = (args.img_height, args.img_width)
    filepaths, labels, class_names = collect_filepaths_and_labels(Path(args.data_dir))

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        filepaths,
        labels,
        test_size=args.validation_split,
        random_state=args.seed,
        stratify=labels,
    )

    train_ds = make_dataset(
        train_paths,
        train_labels,
        image_size=image_size,
        batch_size=args.batch_size,
        training=True,
        seed=args.seed,
    )
    val_ds = make_dataset(
        val_paths,
        val_labels,
        image_size=image_size,
        batch_size=args.batch_size,
        training=False,
        seed=args.seed,
    )

    print(f"Found {len(filepaths)} files belonging to {len(class_names)} classes.")
    print(f"Using {len(train_paths)} files for training.")
    print(f"Using {len(val_paths)} files for validation.")
    print("Training split per class:")
    for class_index, class_name in enumerate(class_names):
        print(f"  {class_name}: {int(np.sum(train_labels == class_index))}")
    print("Validation split per class:")
    for class_index, class_name in enumerate(class_names):
        print(f"  {class_name}: {int(np.sum(val_labels == class_index))}")

    return train_ds, val_ds, class_names


def build_model(
    num_classes: int,
    image_size: tuple[int, int],
    learning_rate: float,
    weights: str,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights=None if weights == "none" else weights,
        input_shape=(image_size[0], image_size[1], 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="vgg16_birads_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def plot_training_curves(history_dict: dict[str, list[float]], output_dir: Path) -> None:
    epochs = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict["accuracy"], label="Train Accuracy")
    plt.plot(epochs, history_dict["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict["loss"], label="Train Loss")
    plt.plot(epochs, history_dict["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    combined: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            combined.setdefault(key, []).extend(values)
    return combined


def evaluate_and_save(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    class_names: list[str],
    output_dir: Path,
) -> None:
    y_true = np.concatenate([labels.numpy() for _, labels in val_ds], axis=0)
    probabilities = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    (output_dir / "classification_report.txt").write_text(report_text + "\n")
    (output_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    np.save(output_dir / "confusion_matrix.npy", cm)
    save_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    print("\nValidation classification report:\n")
    print(report_text)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_counts = count_images_per_class(data_dir)
    if len(class_counts) != 4:
        raise ValueError(
            f"Expected exactly 4 class folders under {data_dir}, found {len(class_counts)}: {sorted(class_counts)}"
        )

    print("Dataset class counts:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    train_ds, val_ds, class_names = build_datasets(args)
    image_size = (args.img_height, args.img_width)
    model, base_model = build_model(len(class_names), image_size, args.learning_rate, args.weights)

    steps_per_epoch = math.ceil(sum(class_counts.values()) * (1.0 - args.validation_split) / args.batch_size)
    validation_steps = math.ceil(sum(class_counts.values()) * args.validation_split / args.batch_size)

    print(f"\nClass names: {class_names}")
    print(f"Approx. steps per epoch: {steps_per_epoch}")
    print(f"Approx. validation steps: {validation_steps}")
    model.summary()

    if args.dry_run:
        print("\nDry run complete. Model and datasets were built successfully.")
        return

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_vgg16_birads.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    print("\nStage 1: training classifier head...")
    history_stage_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history_stage_2 = None
    if args.fine_tune_epochs > 0:
        print("\nStage 2: fine-tuning upper VGG16 layers...")
        base_model.trainable = True
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history_stage_2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=len(history_stage_1.history["loss"]),
            epochs=len(history_stage_1.history["loss"]) + args.fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )

    history_dict = merge_histories(history_stage_1, *( [history_stage_2] if history_stage_2 else [] ))

    model.save(output_dir / "final_vgg16_birads.keras")
    (output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    (output_dir / "history.json").write_text(json.dumps(history_dict, indent=2))
    plot_training_curves(history_dict, output_dir)
    evaluate_and_save(model, val_ds, class_names, output_dir)

    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal validation loss: {val_loss:.4f}")
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
