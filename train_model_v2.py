
import os
import json
import tensorflow as tf

from keras import layers, models, callbacks

# ===== CAU HINH =====
DATA_DIR = "data/raw"
MODEL_PATH = "models/eye_cnn_v2.keras"

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 16
EPOCHS = 40
SEED = 42

tf.keras.utils.set_random_seed(SEED)

os.makedirs("models", exist_ok=True)

# ===== DOC DU LIEU HUAN LUYEN =====
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

# Kiem tra nhan de tranh du doan nguoc
assert train_data.class_names == [
    "closed_eyes",
    "open_eyes"
]

with open(
    "models/eye_cnn_v2_labels.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(train_data.class_names, f)

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)

# ===== TANG CUONG DU LIEU =====
# Chi ap dung khi huan luyen
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.15),
], name="data_augmentation")

# ===== MO HINH CNN V2 =====
model = models.Sequential([
    layers.Input(shape=(96, 96, 3)),

    augmentation,
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.35),

    # closed_eyes = 0, open_eyes = 1
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===== EARLY STOPPING =====
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

# ===== HUAN LUYEN =====
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# ===== DANH GIA VALIDATION =====
loss, accuracy = model.evaluate(
    val_data,
    verbose=0
)

print("\n===== KET QUA CNN V2 =====")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Loss: {loss:.4f}")

# ===== LUU MO HINH MOI =====
model.save(MODEL_PATH)

print("Da luu mo hinh:", MODEL_PATH)
