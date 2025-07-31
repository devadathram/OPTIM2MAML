from Model import UNET, BCEDiceLoss
from sklearn.metrics import accuracy_score
import learn2learn as l2l
from FewShotDataset import PNGSegmentationDataset, SegmentationTaskDataset
import torchvision.transforms as T
from torch import optim
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from ConfigLoader import load_config

# Load config
config = load_config("../configs/traditional_config.yaml")


#Reading the dataset
adc_path = config['dataset']['adc_path']
groundtruth_path = config['dataset']['groundtruth_path']

dataset = PNGSegmentationDataset(adc_path, groundtruth_path)

default_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

#Obtaining the batch of tasks
def get_task_batch(dataset, support_idxs, query_idxs, transform=default_transform):
    support_images, support_masks = [], []
    query_images, query_masks = [], []

    for idx in support_idxs:
        img, mask = dataset[idx]

        #PIL Image before applying transforms
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        img = transform(img)
        mask = transform(mask)
        mask = (mask > 0.5).float()

        support_images.append(img)
        support_masks.append(mask)

    for idx in query_idxs:
        img, mask = dataset[idx]

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        img = transform(img)
        mask = transform(mask)
        mask = (mask > 0.5).float()

        query_images.append(img)
        query_masks.append(mask)

    return (
        torch.stack(support_images),
        torch.stack(support_masks),
        torch.stack(query_images),
        torch.stack(query_masks)
    )

#Model initialisation
model = UNET(in_channels=1, out_channels=1)
maml = l2l.algorithms.MAML(model, lr = config['training']['optimizer']['learning_rate'], first_order=True)
opt = optim.Adam(maml.parameters(), lr=1e-3)
loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
maml = maml.to(device)

#Creating batch of tasks
task_sampler = SegmentationTaskDataset(dataset, k_shot = config['training']['k_shot'], q_query=1, episodes=config['training']['meta_epochs'])

best_loss = float("inf")
accuracy_history = []
episodes = config['training']['meta_epochs']

#MAML Training loop
for episode_idx, (support_idxs, query_idxs) in enumerate(task_sampler):
    print(f"[Episode {episode_idx}] GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    torch.autograd.set_detect_anomaly(True)
    learner = maml.clone()

    support_imgs, support_masks, query_imgs, query_masks = get_task_batch(
        dataset, support_idxs, query_idxs
    )

    support_imgs, support_masks = support_imgs.to(device), support_masks.to(device)
    query_imgs, query_masks = query_imgs.to(device), query_masks.to(device)

    # Inner Loop (Adaptation)
    for _ in range(5):  # 5 gradient steps
        torch.cuda.empty_cache()
        preds = learner(support_imgs)
        loss = loss_fn(preds, support_masks)
        learner.adapt(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            print("yes")
            #torch.save(maml.module.state_dict(), config['training']['save_path'])
            print(f"[Episode {episode_idx}]  New Best Model Saved (Loss: {best_loss:.4f})")

    # Outer Loop (Meta-update)
    query_preds = learner(query_imgs)
    meta_loss = loss_fn(query_preds, query_masks)


    opt.zero_grad()
    meta_loss.backward()
    opt.step()

    #Visualization of the predictions while training
    def visualize_prediction(img, pred_mask, true_mask, idx):
        img = img.squeeze().cpu().numpy()
        pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().detach().numpy()
        true_mask = true_mask.squeeze().cpu().numpy()
        pred_binary = (pred_mask > 0.3).astype(float)
        pred_binary_acc = (pred_mask>0.3).astype(np.float32).flatten()
        true_binary = (true_mask > 0.5).astype(np.float32).flatten()
        accuracy = accuracy_score(true_binary, pred_binary_acc)
        print("ACCURACY: ", accuracy)
        accuracy_history.append(accuracy)

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


    if episode_idx % 50 == 0:
        print(f"[Episode {episode_idx}] Meta Loss: {meta_loss.item():.4f}")
        visualize_prediction(query_imgs[0], query_preds[0], query_masks[0], idx=0)



