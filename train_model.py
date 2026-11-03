
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Cau hinh
DATA_DIR = "data/raw"
MODEL_DIR = "models"
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# Tai anh va chia thanh tap train/validation
train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Thu tu nhan:", train_data.class_names)

# Tang toc doc du lieu
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)

# Xay dung mang CNN
model = models.Sequential([
    layers.Input(shape=(96, 96, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Huan luyen
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Luu mo hinh
model.save("models/eye_cnn.keras")

print("Da luu mo hinh tai models/eye_cnn.keras")
