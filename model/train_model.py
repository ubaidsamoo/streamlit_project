
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Paths
# Adjust these if your folder is named "chest x ray" with spaces
BASE_DIR = 'chest_xray' 
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
VAL_DIR = os.path.join(BASE_DIR, 'val') 

# If your validation folder is named 'vali', uncomment below:
# VAL_DIR = os.path.join(BASE_DIR, 'vali')

# Get absolute path relative to script
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(script_dir, 'model.h5')

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # We use sigmoid for binary classification (Normal vs Pneumonia)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model

def train():
    print(f"Checking directories in {BASE_DIR}...")
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory not found at {TRAIN_DIR}")
        return

    # Data Augmentation for training
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
                                       
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Found training images:")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary') # Binary automatically maps 2 classes to 0 and 1
        
    print("Found validation images:")
    # Use validation set for validation if it exists, otherwise use test
    val_dir_to_use = VAL_DIR if os.path.exists(VAL_DIR) else TEST_DIR
    
    validation_generator = test_datagen.flow_from_directory(
        val_dir_to_use,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary')
        
    # Check class indices
    print("Class Indices:", train_generator.class_indices)
    # Usually {'NORMAL': 0, 'PNEUMONIA': 1}
        
    model = build_model()
    
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )
    
    # Optional: Evaluate on Test set separately if Val was used
    if os.path.exists(TEST_DIR) and val_dir_to_use != TEST_DIR:
        print("Evaluating on separate Test set...")
        test_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False)
        model.evaluate(test_generator)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    print(f"Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("Model saved.")

if __name__ == "__main__":
    train()
