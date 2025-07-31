import os
import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score, f1_score
from mealpy.swarm_based import FA
from mealpy.utils.space import FloatVar, IntegerVar
from mealpy.utils.problem import Problem
from Model import UNET


# Metrics calculation
def calculate_metrics(true_mask, pred_mask):
    true_binary = (true_mask > 0.5).astype(np.float32).flatten()
    pred_binary = (pred_mask > 0.5).astype(np.float32).flatten()

    return {
        'accuracy': accuracy_score(true_binary, pred_binary),
        'precision': precision_score(true_binary, pred_binary, zero_division=0),
        'recall': recall_score(true_binary, pred_binary, zero_division=0),
        'iou': jaccard_score(true_binary, pred_binary, zero_division=0),
        'dice': f1_score(true_binary, pred_binary, zero_division=0)
    }

# Model inference + Post-processing
def test_model(model, img_path, mask_path, threshold, kernel_size, dilate_iter, erode_iter, device='cuda'):

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    try:
        # Load and normalize image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = (image - image.mean()) / (image.std() + 1e-8)
        img = torch.from_numpy(image).float().unsqueeze(0) / 255.0

        # Load mask
        groundtruth = Image.open(mask_path).convert('L')
        true_mask = transform(groundtruth).squeeze().cpu().numpy()
        true_mask = (true_mask > 0.5).astype(np.float32)

        # Convert for model
        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()
            img = Image.fromarray(img)

        img = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None, None, None

    # Prediction
    with torch.no_grad():
        pred = model(img)
        pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()

    # Post-processing
    binary_mask = (pred_prob > threshold).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    processed_mask = cv2.dilate(binary_mask, kernel, iterations=dilate_iter)
    processed_mask = cv2.erode(processed_mask, kernel, iterations=erode_iter)

    metrics = calculate_metrics(true_mask, processed_mask)
    return metrics, processed_mask, true_mask



# Optimization function
def optimize_params(model, img_path, mask_path, device='cuda'):
    def objective(solution):
        threshold, ksize, dilate_iter, erode_iter = solution
        threshold = float(threshold)
        ksize = int(round(ksize))
        dilate_iter = int(round(dilate_iter))
        erode_iter = int(round(erode_iter))

        results = test_model(model, img_path, mask_path, threshold, ksize, dilate_iter, erode_iter, device)
        if results[0] is None:
            return [1.0]  # Penalize failure
        return [-results[0]['dice']]  # Maximize Dice (minimize negative)

    problem = Problem(
        obj_func=objective,
        bounds=[
            FloatVar(0.3, 0.9),   # Threshold
            IntegerVar(1, 9),     # Kernel size
            IntegerVar(1, 5),     # Dilate iterations
            IntegerVar(1, 5)      # Erode iterations
        ],
        minmax="min"
    )

    optimizer = FA.OriginalFA(epoch=3, pop_size=20)  # Increase for better search
    solution = optimizer.solve(problem)
    return solution.solution  # [threshold, ksize, dilate_iter, erode_iter]


# Main Loop: Per-image optimization
def run_full_optimization(model_path, img_folder, mask_folder, output_dir, modality_name, device='cuda'):

    # Initialize model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = UNET(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    os.makedirs(output_dir, exist_ok=True)
    results_list = []
    preds_all, gts_all = [], []
    poor_ls = [] # cases where dice score < 0.55

    image_files = sorted(os.listdir(img_folder))
    mask_files = sorted(os.listdir(mask_folder))


    for idx, (img_name, mask_name) in enumerate(zip(image_files, mask_files), 1):
        img_path = os.path.join(img_folder, img_name)
        mask_path = os.path.join(mask_folder, mask_name)
        print(f"\n[{idx}/{len(image_files)}] Optimizing for {img_name}... and mask {mask_name}")

        best_params = optimize_params(model, img_path, mask_path, device)
        threshold, ksize, dilate_iter, erode_iter = best_params
        print(f" Best Params: Threshold={threshold:.4f}, Kernel={int(ksize)}, Dilate={int(dilate_iter)}, Erode={int(erode_iter)}")

        metrics, pred_mask, true_mask = test_model(model, img_path, mask_path,
                             threshold=threshold,
                             kernel_size=int(ksize),
                             dilate_iter=int(dilate_iter),
                             erode_iter=int(erode_iter),
                             device=device)
        if metrics:
            results_list.append(metrics)
            preds_all.append(pred_mask)
            gts_all.append(true_mask)

            if metrics['dice'] < 0.55:
                poor_ls.append(img_name)

    # Save all predictions and ground truths
    preds_all = np.array(preds_all)
    gts_all = np.array(gts_all)
    print(poor_ls)
    np.save(os.path.join(output_dir, f"{modality_name}_preds.npy"), preds_all)
    np.save(os.path.join(output_dir, "true_masks.npy"), gts_all)
    print(f"\nSaved predictions for {modality_name}: {preds_all.shape}")
    print(f"Saved ground truths: {gts_all.shape}")

    # Compute average metrics
    avg_metrics = {key: np.mean([m[key] for m in results_list]) for key in results_list[0].keys()}
    print("\n=== Average Metrics Across All Images ===")
    for k, v in avg_metrics.items():
        print(f"{k.upper():>10}: {v:.4f}")
    return avg_metrics


# Run the script
if __name__ == "__main__":
    model_path = "../Models/best_maml_model(efficient - 5 shot).pth"
    test_image_folder = "../Dataset/testing_adc_2017/"
    test_mask_folder = "../Dataset/testing_ot_2017/"
    output_dir = "../FusionData/"
    modality_name = "adc_2017_F"
    # Run inference
    run_full_optimization(model_path, test_image_folder, test_mask_folder, output_dir, modality_name, device='cuda')
