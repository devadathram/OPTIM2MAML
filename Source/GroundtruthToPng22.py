import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt


def save_Slice(save_img, name, path):
    if save_img is None or save_img.size == 0:
        print(f"[ERROR] Empty image for {name}. Skipping.")
        return

    if len(save_img.shape) != 2:
        print(f"[ERROR] Unexpected shape {save_img.shape} for {name}. Expected 2D slice. Skipping.")
        return
    os.makedirs(path, exist_ok=True)
    f_saved = os.path.join(path, f"{name}.png")
    save_img = (save_img > 0).astype(np.uint8) * 255
    save_img = cv2.resize(save_img, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    #matplotlib.image.imsave(f_saved, save_img)
    cv2.imwrite(f_saved, save_img)
    print(f'Slice saved: {f_saved}', end='\r')


ot_slices_dict = {}
folder_path = r"../Dataset/derivatives"
save_path_masks = r"../Dataset/ISLES-2022/training_ot"
save_dict_path = r"../Dataset/ISLES-2022/training_ot_slices.npy"

for item in folder_path:
    training_folders = os.path.join(folder_path, item)
    for sub_folder in os.listdir(training_folders):
        if sub_folder.endswith('.json') or sub_folder.endswith('SE') or sub_folder.endswith('.tsv') or sub_folder.endswith('ME') or sub_folder.endswith('.DS_Store'):
            continue
        ses_folder = os.path.join(training_folders, sub_folder)
        for ses in os.listdir(ses_folder):
            if ses.endswith('ore'):
                continue
            msk_folder = os.path.join(ses_folder, ses)
            for msk_file in os.listdir(msk_folder):
                nifty_file = os.path.join(msk_folder, msk_file)
                if os.path.isfile(nifty_file) and nifty_file.endswith(".nii.gz"):
                    img = nib.load(nifty_file)
                    data = img.get_fdata()
                    print(data.shape)
                    slice_indices = [i for i in range(data.shape[2]) if np.any(data[:, :, i] > 0)]
                    print(f"Training folder: {sub_folder}, Slice indices: {slice_indices}")


                    if slice_indices:
                        ot_slices_dict[sub_folder] = slice_indices
                        print(f"Stored {sub_folder}: Slice indices: {slice_indices}")
                        np.save(save_dict_path, ot_slices_dict)

                    for i, slice_idx in enumerate(slice_indices):
                        slice_data = data[:, :, slice_idx]
                        if slice_data is None:
                            print("Onnullado ithil :X")
                        save_Slice(slice_data, f"{sub_folder}_{slice_idx}", save_path_masks)





