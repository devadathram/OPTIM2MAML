import os
import shutil

# Paths
gt_dir = "../Dataset/testing2022/testing_ot"  # Ground truth directory with selected 364 masks
adc_dir = "../Dataset/ISLES-2022/training_new_adc" # ADC images (all saved previously)
adc_target_dir = "../Dataset/testing2022/testing_adc_2022"  # Where to save matched ADCs

# Create the destination folder
os.makedirs(adc_target_dir, exist_ok=True)

# Get list of filenames from groundtruth folder (only .png files)
gt_filenames = [f for f in os.listdir(gt_dir) if f.endswith(".png")]

# Copy corresponding ADC files
copied = 0
for fname in gt_filenames:
    adc_path = os.path.join(adc_dir, fname)
    target_path = os.path.join(adc_target_dir, fname)
    if os.path.exists(adc_path):
        shutil.copy(adc_path, target_path)
        copied += 1
    else:
        print(f"[WARNING] Image not found for: {fname}")

print(f"\n✅ Copied {copied}/{len(gt_filenames)} Images to '{adc_target_dir}'")
