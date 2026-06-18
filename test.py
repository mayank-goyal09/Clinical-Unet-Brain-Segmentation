import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- 1. THE RECIPES (Custom Metrics) ---
def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

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

# --- 3. THE ARCHITECTURE (U-Net) ---
def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(input_shape)
    
    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Bottleneck
    b1 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    b1 = layers.Conv2D(128, 3, activation='relu', padding='same')(b1)
    
    # Decoder
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(b1)
    u1 = layers.concatenate([u1, c1])
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c2)
    
    return models.Model(inputs, outputs)

# --- 4. VISUALIZATION ---
def predict_and_plot(model, generator, num_samples=3):
    X, y_true = generator[0]
    y_pred = model.predict(X)
    
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(min(num_samples, len(X))):
        # Image
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(X[i, :, :, 0], cmap='gray')
        plt.title("MRI Scan")
        plt.axis('off')
        
        # Ground Truth
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(y_true[i, :, :, 0], cmap='gray')
        plt.title("Actual Mask")
        plt.axis('off')
        
        # Prediction
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(y_pred[i, :, :, 0], cmap='gray')
        plt.title("AI Prediction")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# --- 5. EXECUTION ---
if __name__ == "__main__":
    train_gen = BrainDataGenerator('Brain_Data_Clean/images', 'Brain_Data_Clean/masks')
    
    model_path = 'brain_tumor_unet_final.keras'
    
    if os.path.exists(model_path):
        print("🧠 Loading pre-trained model...")
        model = models.load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    else:
        print("🚀 Training model from scratch...")
        model = build_unet()
        model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coef])
        
        callbacks = [
            ModelCheckpoint(model_path, save_best_only=True, monitor='dice_coef', mode='max'),
            EarlyStopping(monitor='dice_coef', patience=10, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='dice_coef', factor=0.5, patience=5, mode='max', min_lr=1e-6)
        ]
        
        model.fit(train_gen, epochs=50, callbacks=callbacks)

    # Show results
    print("📈 Plotting predictions...")
    predict_and_plot(model, train_gen)
