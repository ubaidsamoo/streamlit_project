import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = 'model.h5'
IMG_SIZE = (224, 224)

def check():
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Check 'yes' folder (Should recall high scores ~1.0)
    yes_path = os.path.join('archive', 'yes')
    yes_scores = []
    if os.path.exists(yes_path):
        print("\nChecking YES (Tumor) images:")
        for fname in os.listdir(yes_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(yes_path, fname)
                img = load_img(fpath, target_size=IMG_SIZE)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                score = model.predict(x, verbose=0)[0][0]
                yes_scores.append(score)
                # print(f"{fname}: {score:.4f}")
        
        avg_yes = np.mean(yes_scores) if yes_scores else 0
        print(f"Average Score for YES images (Target > 0.5): {avg_yes:.4f}")
        print(f"Accuracy on YES: {sum(s > 0.5 for s in yes_scores)}/{len(yes_scores)}")

    # Check 'no' folder (Should recall low scores ~0.0)
    no_path = os.path.join('archive', 'no')
    no_scores = []
    if os.path.exists(no_path):
        print("\nChecking NO (Healthy) images:")
        for fname in os.listdir(no_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(no_path, fname)
                img = load_img(fpath, target_size=IMG_SIZE)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                score = model.predict(x, verbose=0)[0][0]
                no_scores.append(score)
        
        avg_no = np.mean(no_scores) if no_scores else 0
        print(f"Average Score for NO images (Target < 0.5): {avg_no:.4f}")
        print(f"Accuracy on NO: {sum(s < 0.5 for s in no_scores)}/{len(no_scores)}")

if __name__ == "__main__":
    check()
