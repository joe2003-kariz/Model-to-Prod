DATA_DIR   = r"C:\Users\Student\OneDrive\Desktop\Data Science\From Model to Production\output"
MODEL_PATH = r"C:\Users\Student\OneDrive\Desktop\Data Science\From Model to Production\TrainedModel"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import json, os

IMG_SIZE   = (128, 128)
BATCH      = 32
EPOCHS     = 5

# 1. Data loaders
train_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2)

train_ds = train_gen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        subset="training")

val_ds   = train_gen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        subset="validation")

base = MobileNetV2(input_shape=(*IMG_SIZE,3), include_top=False, weights="imagenet")
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
outputs = Dense(train_ds.num_classes, activation="softmax")(x)
model = Model(base.input, outputs)

model.compile(optimizer=Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=EPOCHS)

model.save(MODEL_PATH)
print(f"✅  Model saved to {MODEL_PATH}")

class_labels = list(train_ds.class_indices.keys())

LABELS_DIR  = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(LABELS_DIR, exist_ok=True)

LABELS_PATH = os.path.join(LABELS_DIR, "class_labels.json")

with open(LABELS_PATH, "w") as f:
    json.dump(class_labels, f)

print(f"✅ Saved labels to {LABELS_PATH}")
