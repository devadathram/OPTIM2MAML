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


folder_path = r"../Dataset/ISLES-2022"
save_path_masks = r"../Dataset/ISLES-2022/training_ot_dwi"
saved_path_adc = r"../Dataset/ISLES-2022/training_new_adc"
saved_dict_path = r"../Dataset/ISLES-2022/training_ot_slices.npy"

adc_slices = np.load(saved_dict_path, allow_pickle=True).item()

for item in folder_path:
    training_folders = os.path.join(folder_path, item)
    for sub_folder in os.listdir(training_folders):
        if sub_folder.endswith('.json') or sub_folder.endswith('SE') or sub_folder.endswith('.tsv') or sub_folder.endswith('ME') or sub_folder.endswith('csv'):
            continue
        ses_folder = os.path.join(training_folders, sub_folder)
        for ses in os.listdir(ses_folder):
            if ses.endswith('ore'):
                continue
            dwi_folder = os.path.join(ses_folder, ses)
            for dwi in os.listdir(dwi_folder):
                if dwi.endswith('dwi'):
                    adc_folder = os.path.join(dwi_folder, dwi)
                    for adc_file in os.listdir(adc_folder):
                        nifty_file = os.path.join(adc_folder, adc_file)
                        if os.path.isfile(nifty_file) and nifty_file.endswith("adc.nii.gz"):
                            if sub_folder not in adc_slices:
                                print(f"[WARNING] No slice indices found for {sub_folder}, skipping.")
                                continue
                            img = nib.load(nifty_file)
                            data = img.get_fdata()
                            slice_indices = adc_slices[sub_folder]
                            print(f"Processing folder: {sub_folder}, Slice indices: {slice_indices}")

                            for i, slice_idx in enumerate(slice_indices):
                                slice_data = data[:, :, slice_idx]
                                save_Slice(slice_data, f"{sub_folder}_{slice_idx}", saved_path_adc)



