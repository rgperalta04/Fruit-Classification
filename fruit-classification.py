import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.utils import image_dataset_from_directory
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix 

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data/train/train",
                    help="Carpeta con una subcarpeta por clase")
parser.add_argument("--out-dir", default="resultados")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--img-size", type=int, default=160)
parser.add_argument("--epochs-initial", type=int, default=5)
parser.add_argument("--epochs-fine", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--fine-tune-at", type=int, default=130)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (args.img_size, args.img_size)
IMG_SHAPE = IMG_SIZE + (3,)
AUTOTUNE = tf.data.AUTOTUNE

common = dict(batch_size=args.batch_size, image_size=IMG_SIZE,
              validation_split=0.2, seed=args.seed, shuffle=True)
train_dataset = image_dataset_from_directory(args.data_dir, subset="training", **common)
validation_dataset = image_dataset_from_directory(args.data_dir, subset="validation", **common)

class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
print("Clases:", class_names, "| NUM_CLASSES:", NUM_CLASSES, flush=True)
with open(OUT / "labels.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False)

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)


def random_background(images):
    b = tf.shape(images)[0]
    white = tf.cast(tf.reduce_all(images > 250.0, axis=-1, keepdims=True), tf.float32)
    use = tf.cast(tf.random.uniform([b, 1, 1, 1]) < 0.8, tf.float32)
    base = tf.random.uniform([b, 1, 1, 3], 0, 255)
    noise = tf.random.normal(tf.shape(images), stddev=30.0)
    bg = tf.clip_by_value(base + noise, 0, 255)
    m = white * use
    return images * (1.0 - m) + bg * m


aug = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1, fill_mode="reflect"),
    keras.layers.RandomZoom((-0.2, 0.3), fill_mode="reflect"),
    keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
    keras.layers.RandomContrast(0.25),
])


def augment(x, y):
    x = random_background(tf.cast(x, tf.float32))
    x = aug(x, training=True)
    return tf.clip_by_value(x, 0, 255), y


train_aug = train_dataset.map(augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax", name="prediccion")(x)
model = tf.keras.Model(inputs, outputs)
model.summary()


def compile_model(lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])


compile_model(args.lr)
h1 = model.fit(train_aug, validation_data=validation_dataset,
               epochs=args.epochs_initial, verbose=2)

base_model.trainable = True
for layer in base_model.layers[:args.fine_tune_at]:
    layer.trainable = False
compile_model(1e-5)

total_epochs = args.epochs_initial + args.epochs_fine
h2 = model.fit(train_aug, validation_data=validation_dataset,
               epochs=total_epochs, initial_epoch=args.epochs_initial, verbose=2)

history = {k: [float(v) for v in h1.history[k] + h2.history[k]] for k in h1.history}
model.save(OUT / "modelo_fruit.keras")
print("Modelo guardado en", OUT / "modelo_fruit.keras", flush=True)

# ----------------------------------------------------------------------------
# Graficas: accuracy y loss (una figura cada una)
# ----------------------------------------------------------------------------
epochs = np.arange(1, total_epochs + 1)


def plot_metric(key, title, ylabel, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history[key], marker="o", label=f"Train {key}")
    ax.plot(epochs, history[f"val_{key}"], marker="o", label=f"Validation {key}")
    ax.axvline(args.epochs_initial + 0.5, color="gray", linestyle="--",
               label="Inicio fine tuning")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


plot_metric("accuracy", "Accuracy", "Accuracy", "accuracy.png", ylim=(0, 1.05))
plot_metric("loss", "Loss", "Cross entropy", "loss.png")

y_true, y_pred = [], []
for imgs, labels in validation_dataset:
    probs = model.predict_on_batch(imgs)
    y_pred.append(np.argmax(probs, axis=1))
    y_true.append(labels.numpy())
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))


size = max(8, NUM_CLASSES * 0.5)
fig, ax = plt.subplots(figsize=(size, size))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=ax, xticks_rotation=90, colorbar=False,
    include_values=NUM_CLASSES <= 20, values_format="d", cmap="Blues")
ax.set_title("Matriz de confusion (validacion)")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close(fig)

print("Listo. Resultados en:", os.path.abspath(OUT))
