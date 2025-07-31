import os
from PIL import Image
from Model import UNET, BCEDiceLoss
from TaskSampling import PNGSegmentationDataset
import numpy as np
import torch
from torch import nn, optim
import torchvision.transforms as T
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from ConfigLoader import load_config

# Load config
config = load_config("../configs/traditional_config.yaml")

# Paths
adc_path = config['unet_paths']['adc_path']
groundtruth_path = config['unet_paths']['groundtruth_path']

# Hyperparameters
batch_size = 8
epochs = 200
lr = 0.01

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = UNET(in_channels=1, out_channels=1).to(device)

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    #T.Normalize(mean=[0.5], std=[0.5])
])

dataset = PNGSegmentationDataset(adc_path, groundtruth_path)
images_x = []
masks_y = []

# Loss and Optimizer
loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
best_loss = float("inf")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for idx in range(len(os.listdir(adc_path))):
        img, mask = dataset[idx]

        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()
            img = Image.fromarray(img)
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
            mask = Image.fromarray(mask)

        img = transform(img).unsqueeze(0).to(device)
        mask = transform(mask).unsqueeze(0).to(device)
        mask = (mask > 0.5).float()

        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, mask)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataset)
    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        #torch.save(model.state_dict(), config['unet_paths']['save_model_path'])
        print(f"--> New best model saved! Loss: {best_loss:.4f}")

    # Optional: Visualization (one sample)
    def visualize_prediction(img, pred_mask, true_mask, idx):
        img = img.squeeze().cpu().numpy()
        pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().detach().numpy()
        true_mask = true_mask.squeeze().cpu().numpy()

        pred_binary = (pred_mask > 0.3).astype(float)
        pred_binary_acc = (pred_mask > 0.3).astype(np.float32).flatten()
        true_binary = (true_mask > 0.5).astype(np.float32).flatten()
        print("ACCURACY: ", accuracy_score(true_binary, pred_binary_acc))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Predicted Mask")
        plt.imshow(pred_binary, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(true_mask, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


    if epoch % 20 == 0:
        test_img, test_mask = dataset[0]
        if not isinstance(test_img, torch.Tensor):
            test_img = transform(test_img)
        if not isinstance(test_mask, torch.Tensor):
            test_mask = transform(test_mask)

        test_img = test_img.unsqueeze(0).to(device)
        test_mask = test_mask.unsqueeze(0).to(device)
        pred = model(test_img)
        visualize_prediction(test_img[0], pred[0], test_mask[0], idx=0)

