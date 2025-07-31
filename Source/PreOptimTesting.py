import torch
import os
import torchvision.transforms as T
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from Model import UNET
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score, f1_score, auc, roc_curve


# Metrics function
def calculate_metrics(true_mask, pred_mask):
    true_binary = (true_mask > 0.5).astype(np.float32).flatten()
    pred_binary = (pred_mask > 0.5).astype(np.float32).flatten()

    metrics = {
        'accuracy': accuracy_score(true_binary, pred_binary),
        'precision': precision_score(true_binary, pred_binary, zero_division=0),
        'recall': recall_score(true_binary, pred_binary, zero_division=0),
        'iou': jaccard_score(true_binary, pred_binary, zero_division=0),
        'dice': f1_score(true_binary, pred_binary, zero_division=0)
    }
    return metrics

def test_model(model_path, test_image_folder, groundtruth_folder, output_dir, modality_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load Model
    model = UNET(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Define transforms
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        #T.Normalize(mean=[0.5], std=[0.5])
    ])

    image_files = sorted(os.listdir(test_image_folder))
    mask_files = sorted(os.listdir(groundtruth_folder))
    results_list = []
    preds_all = []
    gts_all =[]

    for idx, (img_name, mask_name) in enumerate(zip(image_files, mask_files), 1):
        image = os.path.join(test_image_folder, img_name)
        mask = os.path.join(groundtruth_folder, mask_name)
        print(f"\n[{idx}/{len(image_files)}] Optimizing for {img_name}... and mask {mask_name}")

        # Preprocess
        try:
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (256,256))
            image = (image - image.mean()) / (image.std() + 1e-8)
            img = torch.from_numpy(image).float().unsqueeze(0) / 255.0

            groundtruth = Image.open(mask).convert('L')
            true_mask = transform(groundtruth).squeeze().cpu().numpy()  # No sigmoid for ground truth
            true_mask = (true_mask > 0.5).astype(np.float32)

            if isinstance(img, torch.Tensor):
                img = img.squeeze().cpu().numpy()
                img = Image.fromarray(img)

            img = transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading images: {e}")
            return None

        # Make prediction
        with torch.no_grad():
            pred = model(img)
            pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
            preds_all.append(pred_prob)
            gts_all.append(true_mask)


        # Post-processing
        binary_mask = (pred_prob > 0.3).astype(np.uint8)

        kernel = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=3)
        dilated_mask = cv2.erode(dilated_mask, kernel, iterations=1)


        metrics = calculate_metrics(true_mask, dilated_mask)
        results_list.append(metrics)
        print("\nSegmentation Metrics:")
        print("-" * 30)
        for name, value in metrics.items():
            print(f"{name.upper():>10}: {value:.4f}")
        print("-" * 30)

    preds_all = np.array(preds_all)
    gts_all = np.array(gts_all)
    np.save(os.path.join(output_dir, f"{modality_name}_preds.npy"), preds_all)
    np.save(os.path.join(output_dir, "true_masks.npy"), gts_all)
    print(f"\nSaved predictions for {modality_name}: {preds_all.shape}")
    print(f"Saved ground truths: {gts_all.shape}")


    # Calculate average metrics across  the dataset
    avg_metrics = {key: np.mean([m[key] for m in results_list]) for key in results_list[0].keys()}
    print("\n=== Average Metrics Across All Images ===")
    for k, v in avg_metrics.items():
        print(f"{k.upper():>10}: {v:.4f}")
    return avg_metrics


if __name__ == "__main__":
    model_path = "../Models/best_maml_model(efficient - 5 shot 2022 - dwi).pth"
    test_img_folder = "../Dataset/testing_dwi_new_2022/"
    test_gt_folder = "../Dataset/testing_ot/"
    output_dir = "../FusionData/"
    modality = "dwi_preoptim"
    # Run inference
    results = test_model(model_path, test_image_folder=test_img_folder, groundtruth_folder=test_gt_folder, output_dir = output_dir, modality_name=modality)