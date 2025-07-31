import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.measure import label, regionprops


# Metrics Function
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

# Fusion Function
def score_level_fusion_smoothed(adc_preds, dwi_preds, true_masks, weights=(0.5, 0.5), threshold=0.5, sigma=1.0):
    fused_probs = weights[0] * adc_preds + weights[1] * dwi_preds
    fused_probs = gaussian(fused_probs, sigma=sigma, preserve_range=True)
    fused_mask = (fused_probs > threshold).astype(np.uint8)
    metrics_list = [calculate_metrics(true_masks[i], fused_mask[i]) for i in range(len(true_masks))]
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
    return fused_mask, avg_metrics

# Largest Component Filter
def keep_largest_component(mask):
    new_mask = []
    for i in range(mask.shape[0]):
        lbl = label(mask[i])
        regions = regionprops(lbl)
        if not regions:
            new_mask.append(mask[i])
            continue
        largest = max(regions, key=lambda r: r.area)
        largest_mask = (lbl == largest.label).astype(np.uint8)
        new_mask.append(largest_mask)
    return np.stack(new_mask)

# Visualization
def visualize_fusion(pred_mask, true_masks, idx=0):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Fused Prediction")
    plt.imshow(pred_mask[idx], cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Ground Truth")
    plt.imshow(true_masks[idx], cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Main function
if __name__ == "__main__":
    # Load predictions and ground truth
    adc_preds = np.load("../FusionData/adc_preds.npy")
    dwi_preds = np.load("../FusionData/dwi_preds.npy")
    true_masks = np.load("../FusionData/true_masks_2022.npy")

    print("ADC range:", adc_preds.min(), adc_preds.max())
    print("DWI range:", dwi_preds.min(), dwi_preds.max())
    if adc_preds.max() <= 1 and dwi_preds.max() <= 1:
        print("Predictions appear normalized.")
    else:
        print("Warning: Predictions may need normalization.")

    # Search best weight, threshold, and smoothing sigma
    best_dice = 0
    best_params = (0.5, 0.5, 0.5, 1.0)
    for w in np.arange(0.0, 1.05, 0.05):
        for t in np.arange(0.1, 0.9, 0.02):
            for sigma in [0.5, 1.0, 1.5]:
                fused_mask, metrics = score_level_fusion_smoothed(adc_preds, dwi_preds, true_masks,
                                                                  weights=(w, 1 - w), threshold=t, sigma=sigma)
                if metrics['dice'] > best_dice:
                    best_dice = metrics['dice']
                    best_params = (w, 1 - w, t, sigma)

    print("\n=== Best Fusion Parameters ===")
    print(f"Weights: {best_params[0]:.2f}, {best_params[1]:.2f}")
    print(f"Threshold: {best_params[2]:.2f}")
    print(f"Sigma: {best_params[3]:.2f}")
    print(f"Best Dice: {best_dice:.4f}")

    # Apply best fusion
    fused_mask, _ = score_level_fusion_smoothed(adc_preds, dwi_preds, true_masks,
                                                weights=(best_params[0], best_params[1]),
                                                threshold=best_params[2],
                                                sigma=best_params[3])

    # Postprocessing: Keep only largest connected component
    fused_mask_post = keep_largest_component(fused_mask)

    # Final evaluation
    metrics_post = [calculate_metrics(true_masks[i], fused_mask_post[i]) for i in range(len(true_masks))]
    final_metrics = {k: np.mean([m[k] for m in metrics_post]) for k in metrics_post[0].keys()}
    print("\n=== Final Metrics After Postprocessing ===")
    for k, v in final_metrics.items():
        print(f"{k.upper():>10}: {v:.4f}")

    # Save result
    np.save("../FusionData/fused_best_smoothed_postprocessed.npy", fused_mask_post)

    # Visualize results
    visualize_fusion(fused_mask_post, true_masks, idx=0)
