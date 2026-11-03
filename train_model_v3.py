
import os
import tensorflow as tf
from keras import layers, models, callbacks

# ===== CAU HINH =====
DATA_DIR = "data/v3_train"
MODEL_DIR = "models"

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 16
EPOCHS = 40
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# ===== TAI DU LIEU =====
train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

print("Thu tu nhan:", train_data.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)

# ===== TANG CUONG DU LIEU =====
augmentation = models.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.15),
], name="augmentation")

# ===== XAY DUNG CNN V3 =====
model = models.Sequential([
    layers.Input(shape=(96, 96, 3)),

    augmentation,

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.35),

    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ===== CALLBACKS =====
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001,
)

# ===== HUAN LUYEN =====
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
)

# ===== LUU MO HINH =====
model.save("models/eye_cnn_v3.keras")

print("\nDa luu CNN V3: models/eye_cnn_v3.keras")
