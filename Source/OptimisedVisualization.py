import torch
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

# Visualizing a single image - can be used for Pre-Optim + MAML as well as Post-Optim (apply parameters accordingly)
def test_model(model_path, test_image_path, groundtruth_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
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

    # Preprocess
    try:
        image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256,256))
        image = (image - image.mean()) / (image.std() + 1e-8)
        img = torch.from_numpy(image).float().unsqueeze(0) / 255.0

        groundtruth = Image.open(groundtruth_path).convert('L')
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


    # Post-processing
    binary_mask = (pred_prob > 0.5).astype(np.uint8)

    kernel = np.ones((3,3), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=3)
    dilated_mask = cv2.erode(dilated_mask, kernel, iterations=1)


    metrics = calculate_metrics(true_mask, dilated_mask)
    print("\nSegmentation Metrics:")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"{name.upper():>10}: {value:.4f}")
    print("-" * 30)


    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    # Input Image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Probability Mask
    prob_plot = axes[1].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Probability Mask")
    plt.colorbar(prob_plot, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')

    # Binary Prediction
    axes[2].imshow(dilated_mask, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (Dice: {metrics['dice']:.3f})")
    axes[2].axis('off')

    #Ground truth
    axes[3].imshow(true_mask, cmap='gray')
    axes[3].set_title("Ground truth")
    axes[3].axis('off')


    plt.tight_layout()
    plt.show()

    return {
        'image': np.array(image),
        'prob_mask': pred_prob,
        'binary_mask': binary_mask,
        'dilated_mask': dilated_mask
    }


if __name__ == "__main__":
    model_path = "../Models/best_maml_model(efficient - 3 shot).pth"
    test_image_path = "../Dataset/testing2017/testing_adc_2017/training_4_VSD_11.png"
    test_image_groundtruth = "../Dataset/testing2017/testing_ot_2017/training_4_VSD_11.png"
    # Run inference
    results = test_model(model_path, test_image_path, test_image_groundtruth)