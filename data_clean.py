import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_IMG_DIR = 'Task01_BrainTumour/imagesTr' # Update this to your path
RAW_LABEL_DIR = 'Task01_BrainTumour/labelsTr'
OUTPUT_DIR = 'Brain_Data_Clean'

# Create output folders
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/masks", exist_ok=True)

def process_dataset():
    # Get list of training files
    image_files = [f for f in os.listdir(RAW_IMG_DIR) if not f.startswith('.')]
    
    print(f"🧠 Found {len(image_files)} patients. Starting distillation...")

    for file_name in tqdm(image_files):
        # 1. Load 4D Image and 3D Mask
        img_path = os.path.join(RAW_IMG_DIR, file_name)
        mask_path = os.path.join(RAW_LABEL_DIR, file_name)
        
        # Load data as numpy arrays
        img_obj = nib.load(img_path).get_fdata()
        mask_obj = nib.load(mask_path).get_fdata()

        # 2. Extract FLAIR (Channel 0 in MSD Task01)
        # MSD format is [H, W, Slices, Channels]
        flair_data = img_obj[:, :, :, 0] 

        # 3. Iterate through slices
        num_slices = flair_data.shape[2]
        for s in range(num_slices):
            slice_mask = mask_obj[:, :, s]
            
            # ONLY save if the slice has a significant tumor (more than 10 pixels)
            if np.sum(slice_mask) > 10:
                slice_img = flair_data[:, :, s]
                
                # Normalize slice to 0-1 range
                if np.max(slice_img) > 0:
                    slice_img = slice_img / np.max(slice_img)

                # 4. Resize to 128x128 to save massive space
                resized_img = cv2.resize(slice_img, (128, 128), interpolation=cv2.INTER_AREA)
                resized_mask = cv2.resize(slice_mask, (128, 128), interpolation=cv2.INTER_NEAREST)

                # 5. Save as lightweight .npy files
                save_name = f"{file_name.split('.')[0]}_slice{s}"
                np.save(f"{OUTPUT_DIR}/images/{save_name}.npy", resized_img.astype(np.float32))
                np.save(f"{OUTPUT_DIR}/masks/{save_name}.npy", resized_mask.astype(np.uint8))

    print(f"✅ Success! Your clean data is in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    process_dataset()