import torch
import os
import torchvision.transforms as T
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from Model import UNET
import numpy as np


#metrics calculation
def calculate_metrics(true_mask, pred_mask):
    true_binary = (true_mask > 0.5).astype(np.float32).flatten()
    pred_binary = (pred_mask > 0.92).astype(np.float32).flatten()

    metrics = {
        'accuracy': accuracy_score(true_binary, pred_binary),
        'precision': precision_score(true_binary, pred_binary, zero_division=0),
        'recall': recall_score(true_binary, pred_binary, zero_division=0),
        'iou': jaccard_score(true_binary, pred_binary, zero_division=0),
        'dice': f1_score(true_binary, pred_binary, zero_division=0)
    }
    return metrics


def test_model(model_path, test_image_folder, groundtruth_folder, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load model
    try:
        model = UNET(in_channels=1, out_channels=1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Define transforms
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    image_files = sorted(os.listdir(test_image_folder))
    mask_files = sorted(os.listdir(groundtruth_folder))
    results_list = []

    for idx, (img_name, mask_name) in enumerate(zip(image_files, mask_files), 1):
        image = os.path.join(test_image_folder, img_name)
        mask = os.path.join(groundtruth_folder, mask_name)
        print(f"\n[{idx}/{len(image_files)}] Optimizing for {img_name}... and mask {mask_name}")

    # Load and preprocess images
        try:
            img = Image.open(image).convert('L')
            img_tensor = transform(img).unsqueeze(0).to(device)

            groundtruth = Image.open(mask).convert('L')
            true_mask = transform(groundtruth).squeeze().cpu().numpy()  # No sigmoid for ground truth
            true_mask = (true_mask > 0.5).astype(np.float32)  # Binarize ground truth
        except Exception as e:
            print(f"Error loading images: {e}")
            return None

        # Make prediction
        with torch.no_grad():
            pred = model(img_tensor)
        prob_mask = torch.sigmoid(pred.squeeze()).cpu().numpy()
        binary_mask = (prob_mask > 0.3).astype(np.float32)  # Standard threshold
        kernel = np.ones((3, 3))
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=3)
        dilated_mask = cv2.erode(dilated_mask, kernel, iterations=1)

        # Calculate and print metrics
        metrics = calculate_metrics(true_mask, dilated_mask)
        results_list.append(metrics)
        print("\nSegmentation Metrics:")
        print("-" * 30)
        for name, value in metrics.items():
            print(f"{name.upper():>10}: {value:.4f}")
        print("-" * 30)


    # Calculate average metrics
    avg_metrics = {key: np.mean([m[key] for m in results_list]) for key in results_list[0].keys()}
    print("\n=== Average Metrics Across All Images ===")
    for k, v in avg_metrics.items():
        print(f"{k.upper():>10}: {v:.4f}")
    return avg_metrics


if __name__ == "__main__":
    model_path = "../Models/best_maml_model(3 shot).pth"
    test_img_folder = "../Dataset/testing_adc_2017/"
    test_gt_folder = "../Dataset/testing_ot_2017/"
    # Run inference
    results = test_model(model_path, test_image_folder=test_img_folder, groundtruth_folder=test_gt_folder)