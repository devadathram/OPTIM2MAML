import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np
from torch.ao.nn.quantized.functional import threshold

from Model import UNET
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score, f1_score
from mealpy.swarm_based import PSO
from mealpy.utils.space import FloatVar, IntegerVar
from mealpy.utils.problem import  Problem


# Metrics function
def calculate_metrics(true_mask, pred_mask, threshold=0.92):
    true_binary = (true_mask > 0.5).astype(np.float32).flatten()
    pred_binary = (pred_mask > threshold).astype(np.float32).flatten()

    metrics = {
        'accuracy': accuracy_score(true_binary, pred_binary),
        'precision': precision_score(true_binary, pred_binary, zero_division=0),
        'recall': recall_score(true_binary, pred_binary, zero_division=0),
        'iou': jaccard_score(true_binary, pred_binary, zero_division=0),
        'dice': f1_score(true_binary, pred_binary, zero_division=0)
    }
    return metrics


# Function for optimizing a single image
def test_model(model_path, test_image_path, groundtruth_path,
               threshold=0.92, kernel_size=1, dilate_iter=2, erode_iter=2,
               device='cuda' if torch.cuda.is_available() else 'cpu',
               verbose=True):

    # Load model
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
    binary_mask = (pred_prob > threshold).astype(np.uint8)  # Use same threshold as training

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=dilate_iter)
    dilated_mask = cv2.erode(dilated_mask, kernel, iterations=erode_iter)


    metrics = calculate_metrics(true_mask, dilated_mask, threshold)
    print("\nSegmentation Metrics:")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"{name.upper():>10}: {value:.4f}")
    print("-" * 30)

    return metrics

# Define optimization function
def firefly_optimize(model_path, test_image_path, groundtruth_path):
    def objective_function(solution):
        threshold, ksize, dilate_iter, erode_iter = solution
        threshold = float(threshold)
        ksize = int(round(ksize))
        dilate_iter = int(round(dilate_iter))
        erode_iter = int(round(erode_iter))

        try:
            results = test_model(
                model_path=model_path,
                test_image_path=test_image_path,
                groundtruth_path=groundtruth_path,
                threshold=threshold,
                kernel_size=ksize,
                dilate_iter=dilate_iter,
                erode_iter=erode_iter,
                verbose=False  # turn off plotting during optimization
            )
            return [-results['dice']]  # minimize negative Dice
        except Exception as e:
            print("Error during optimization eval:", e)
            return [1.0]  # Worst possible score

    problem = Problem(
        obj_func=objective_function,
        bounds=[
            FloatVar(0.1, 0.9),
            IntegerVar(1, 10),
            IntegerVar(1, 5),
            IntegerVar(1, 5),
        ],
        minmax="min"
    )

    model = PSO.OriginalPSO(epoch=5, pop_size=20)
    solution = model.solve(problem)
    best_solution = solution.solution
    best_fitness = solution.target.fitness
    print("\nBest Post-processing Parameters Found:")
    print(f"Threshold: {best_solution[0]:.4f}")
    print(f"Kernel Size: {int(best_solution[1])}")
    print(f"Dilate Iter: {int(best_solution[2])}")
    print(f"Erode Iter: {int(best_solution[3])}")
    print(f"Best fitness: {best_fitness}")
    return best_solution


model_path = "../Models/best_maml_model(efficient - 5 shot 2022 - adc).pth"
test_image_path = "../Dataset/ISLES-2022/training_new_adc/sub-strokecase0031_41.png"
test_ot_path = "../Dataset/ISLES-2022/training_ot/sub-strokecase0031_41.png"
# Run inference
best_params = firefly_optimize(model_path, test_image_path, test_ot_path)
final_results = test_model(
        model_path, test_image_path, test_ot_path,
        threshold=best_params[0],
        kernel_size=int(round(best_params[1])),
        dilate_iter=int(round(best_params[2])),
        erode_iter=int(round(best_params[3])),
        verbose=True
    )