import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, backend as K
from tensorflow.keras.utils import Sequence

# Set console encoding to UTF-8 on Windows to avoid UnicodeEncodeError when printing emojis
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for Python versions that don't support reconfigure (though 3.13 does)
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# --- 1. THE RECIPES (Custom Metrics) ---
def dice_coef(y_true, y_pred):
    smooth = 1e-6
    
    # Force everything to float32 and clip to [0, 1] range
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # This is the "Safety Guard": ensures no pixel is > 1.0
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    return score

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# --- 2. THE CONVEYOR BELT (Data Generator) ---
class BrainDataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=16, img_size=128):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.image_files = sorted(os.listdir(image_dir))
        
    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, idx):
        batch_files = self.image_files[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = np.empty((self.batch_size, self.img_size, self.img_size, 1), dtype=np.float32)
        y = np.empty((self.batch_size, self.img_size, self.img_size, 1), dtype=np.float32)

        for i, file_name in enumerate(batch_files):
            img = np.load(os.path.join(self.image_dir, file_name))
            mask = np.load(os.path.join(self.mask_dir, file_name)) 
            X[i, :, :, 0] = img
            y[i, :, :, 0] = mask
        return X, y

if __name__ == "__main__":
    model_path = 'brain_tumor_unet_final.keras'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found. Please train the model first by running test.py")
    else:
        print("🧠 Loading model...")
        model = models.load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
        
        print("📁 Loading data...")
        eval_gen = BrainDataGenerator('Brain_Data_Clean/images', 'Brain_Data_Clean/masks', batch_size=1)
        
        scores = []
        num_tests = 50
        print(f"📊 Calculating final scores for {num_tests} random slices...")

        for i in range(num_tests):
            # Pick a random batch (batch_size=1)
            idx = np.random.randint(0, len(eval_gen))
            X, y = eval_gen[idx]
            
            pred = model.predict(X, verbose=0)
            
            # Use tf.cast to ensure types match for the metric
            score = dice_coef(tf.cast(y, tf.float32), tf.cast(pred, tf.float32))
            scores.append(score.numpy())
            
            if (i+1) % 10 == 0:
                print(f"✅ Processed {i+1}/{num_tests}...")

        print("\n" + "="*40)
        print(f"🔥 Average Dice Score: {np.mean(scores):.4f}")
        print("✅ Project complete. You are officially a Medical AI Developer, Mayank!")
        print("="*40)
