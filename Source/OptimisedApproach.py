import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, roc_curve
from TaskSampling import TaskSampler, PNGSegmentationDataset
from Model import UNET, BCEDiceLoss
import learn2learn as l2l
from torch import optim
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.mixture import GaussianMixture
from PIL import Image
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ConfigLoader import load_config

# Load config
config = load_config("../configs/optimised_config.yaml")


#Reading the dataset
adc_path = config['dataset']['adc_path']
groundtruth_path = config['dataset']['groundtruth_path']
num_tasks = config['clustering']['num_tasks']

#Class Initialisation
dataset = PNGSegmentationDataset(adc_path, groundtruth_path)
sampler = TaskSampler(image_dir=adc_path, mask_dir=groundtruth_path, device='cuda')
clustered_tasks = sampler.cosine_cluster_tasks(num_tasks = num_tasks, metric='cosine')
model = UNET(in_channels=1, out_channels=1)
maml = l2l.algorithms.MAML(model, lr=0.01, first_order=True)

optimizer_type = config['training']['optimizer']['type'].lower()
lr = config['training']['optimizer']['lr']

if optimizer_type == 'adam':
    opt = torch.optim.Adam(model.parameters(), lr=lr)
elif optimizer_type == 'sgd':
    opt = torch.optim.SGD(model.parameters(), lr=lr)

loss_fn = BCEDiceLoss(bce_weight=config['training']['loss_fn']['bce_weight'], dice_weight=config['training']['loss_fn']['dice_weight'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
maml = maml.to(device)
episodes = config['training']['episodes']

# Transforms
transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    #T.Normalize(mean=[0.5], std=[0.5])
])

best_loss = float("inf")
k_shots = config['dataset']['k_shots']
k_query = 1


#MAML Training loop
for episode_idx in range(episodes + 1):
    cluster_keys = list(clustered_tasks.keys())
    cluster_idx = episode_idx % len(cluster_keys)
    cluster = clustered_tasks[cluster_keys[cluster_idx]]
    start_idx = (episode_idx // len(cluster_keys) * k_shots) % len(cluster)
    task_ids = [cluster[(start_idx + i) % len(cluster)] for i in range(k_shots + k_query)]
    learner = maml.clone()

    support_imgs, support_masks, query_imgs, query_masks = [], [], [], []

    support_ids, query_ids = sampler.sample_task_images(task_ids, k_support=k_shots, k_query=k_query)

    for idx in support_ids:
        img, mask = dataset[idx]
        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()
            img = Image.fromarray(img)
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
            mask = Image.fromarray(mask)

        img = transform(img)
        mask = transform(mask)
        mask = (mask > 0.5).float()
        support_imgs.append(img)
        support_masks.append(mask)

    for idx in query_ids:
        img, mask = dataset[idx]
        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()  # Convert tensor to ndarray
            img = Image.fromarray(img)  # Convert ndarray to PIL Image if needed
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
            mask = Image.fromarray(mask)

        img = transform(img)  # Apply transform (resize and ToTensor)
        mask = transform(mask)  # Apply transform (resize and ToTensor)

        mask = (mask > 0.5).float()  # Binarize mask

        query_imgs.append(img)
        query_masks.append(mask)

    # Stack the images and masks
    support_imgs = torch.stack(support_imgs).to(device)
    support_masks = torch.stack(support_masks).to(device)

    query_imgs = torch.stack(query_imgs).to(device)
    query_masks = torch.stack(query_masks).to(device)


    # Inner Loop (Adaptation)
    for _ in range(5):  # 5 gradient steps
        torch.cuda.empty_cache()
        preds = learner(support_imgs)
        loss = loss_fn(preds, support_masks)
        learner.adapt(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            #torch.save(model.state_dict(), config['training']['save_path'])
            print(f"[Episode {episode_idx}]  New Best Model Saved (Loss: {best_loss:.4f})")

    # Outer Loop (Meta-update)
    query_preds = learner(query_imgs)
    meta_loss = loss_fn(query_preds, query_masks)

    opt.zero_grad()
    meta_loss.backward()
    opt.step()

    #Visualization
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



    if episode_idx % 50 == 0:
        print(f"[Episode {episode_idx}] Meta Loss: {meta_loss.item():.4f}")
        visualize_prediction(query_imgs[0], query_preds[0], query_masks[0], idx=0)


'''
#clustered_tasks = sampler.gmm_clustering()
#clustered_tasks = sampler.cluster_tasks(num_tasks=3)
extracted_features = sampler.extract_features()


#Cluster visualisation
def visualize_clusters(features, extracted_features, cluster_labels, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    extracted_new_features = reducer.fit_transform(extracted_features)
    reduced_features = reducer.fit_transform(features)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(extracted_new_features[:, 0], extracted_new_features[:, 1], s=30)
    plt.title(f'Before Pre-Optim ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(scatter)
    plt.show()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='tab10', s=30)
    plt.title(f'After Pre-Optim ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(scatter, ticks=range(len(set(cluster_labels))))
    plt.show()

#Clustered Image visualisation

def visualize_clustered_tasks(clustered_tasks, num_images=3):
    for task_id, image_paths in clustered_tasks.items():
        print(f"\nTask {task_id}: showing up to {num_images} images")
        plt.figure(figsize=(15, 3))
        for i, img_path in enumerate(image_paths[:num_images]):
            image = Image.open(img_path).convert('L')  # Grayscale
            plt.subplot(1, num_images, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(os.path.basename(img_path))
            plt.axis('off')
        plt.suptitle(f"GMM Cluster {task_id}")
        plt.tight_layout()
        plt.show()


visualize_clustered_tasks(clustered_tasks, num_images=4)


#kmeans = KMeans(n_clusters=3, random_state=42)
#cluster_labels = kmeans.fit_predict(sampler.features)

similarity = sampler.compute_similarity(metric='cosine')
spectral_clustering = SpectralClustering(n_clusters=3, affinity='precomputed', random_state=42)
cluster_labels = spectral_clustering.fit_predict(similarity)

#gmm_clustering = GaussianMixture(n_components=3, random_state=42)
#cluster_labels = gmm_clustering.fit_predict(sampler.features)


# Visualize clusters
#visualize_clusters(sampler.features, extracted_features=sampler.features, cluster_labels=cluster_labels, method='pca')

'''

