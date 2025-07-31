import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.image


def save_Slice(save_img, name, path):
    os.makedirs(path, exist_ok=True)
    f_saved = os.path.join(path, f"{name}.png")
    save_img = (save_img - np.min(save_img)) / (np.max(save_img) - np.min(save_img) + 1e-8)
    save_img = (save_img * 255).astype(np.uint8)
    save_img = cv2.resize(save_img, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    #matplotlib.image.imsave(f_saved, save_img)
    cv2.imwrite(f_saved, save_img)
    print(f'Slice saved: {f_saved}', end='\r')


folder_path = r"../Dataset\ISLES2017\testing"
save_path_masks = r"../Dataset\ISLES2017\testing_ot"
save_path_dpwi = r"../Dataset\ISLES2017\testing_adc"
save_dict_path = r"../Dataset\ISLES2017\testing_adc_slices.npy"

adc_slices = np.load(save_dict_path, allow_pickle=True).item()

for item in os.listdir(folder_path):
    training_folders = os.path.join(folder_path, item)
    for file in os.listdir(training_folders):
        VSD_folders = os.path.join(training_folders, file)

        patient_name = item

        dpwi_folder = [f for f in os.listdir(VSD_folders) if "MR_ADC" in f]

        for dpwi_file in dpwi_folder:
            nifty_file = os.path.join(VSD_folders, dpwi_file)
            if os.path.isfile(nifty_file) and nifty_file.endswith(".nii.gz"):
                img = nib.load(nifty_file)
                data = img.get_fdata()
                slice_indices = adc_slices[patient_name]
                print(f"Processing folder: {training_folders}, Slice indices: {slice_indices}")

                for i, slice_idx in enumerate(slice_indices):
                    slice_data = data[:, :, slice_idx]
                    save_Slice(slice_data, f"{patient_name}_{file}_{slice_idx}", save_path_dpwi)



