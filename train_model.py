import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---
DATASET_PATH = 'archive'  # Pointing to the directory containing 'yes' and 'no' folders
MODEL_SAVE_PATH = 'model.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

def build_model():
    """
    Builds the CNN model using MobileNetV2 as the base.
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Unfreeze the last 20 layers of the base model for Fine-Tuning
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Increased dropout slightly
        Dense(1, activation='sigmoid')
    ])
    
    # Lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-4), 
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train():
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' not found.")
        return

    # Data Augmentation
    # NOTE: MobileNetV2 expects specific preprocessing [-1, 1], not just 1/255.
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% validation split
    )

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=['no', 'yes'],
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    print("Loading Validation Data...")
    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=['no', 'yes'],
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    if train_generator.samples == 0:
        print("No images found! Please check the dataset structure.")
        return

    # Build Model
    model = build_model()
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Train
    print("Starting Training (No Class Weights)...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint]
    )

    # Save final model if not saved by checkpoint
    if not os.path.exists(MODEL_SAVE_PATH):
        model.save(MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

    # Plot Results
    plot_history(history)

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved as training_history.png")

if __name__ == "__main__":
    train()
