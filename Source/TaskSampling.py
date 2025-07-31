import os
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from torchvision import transforms
from PIL import Image
import random
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PNGSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        #self.mask_filenames = [f.replace('.png', '_mask.png') for f in self.image_filenames]
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load image using OpenCV and apply normalisation
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0  # [1, H, W]

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

#CNN to extract features for clustering
class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 63 * 63, 128)
        )

    def forward(self, x):
        return self.encoder(x)


class TaskSampler:
    def __init__(self, image_dir: str, mask_dir: str, encoder=DummyEncoder(),
                 device: str = 'cuda', transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder.to(self.device).eval()
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128))
        ])
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith('.png')
        ])
        self.dataset = PNGSegmentationDataset(self.image_dir, self.mask_dir, transform=self.transform, mask_transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)
        self.features = self.extract_features()

    # Extracting features from images
    def extract_features(self):
        features = []
        with torch.no_grad():
            for batch, _ in self.dataloader:
                batch = batch.to(self.device)
                out = self.encoder(batch)
                features.append(out.cpu().numpy())
        return np.vstack(features)

    # Similarity computation for spectral clustering
    def compute_similarity(self, metric='cosine'):
        if metric == 'cosine':
            print(self.features.shape)
            return cosine_similarity(self.features)
        elif metric == 'euclidean':
            return -pairwise_distances(self.features, metric='euclidean')  # negative for similarity
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    # K-Means
    def cluster_tasks(self, num_tasks=5):
        kmeans = KMeans(n_clusters=num_tasks, random_state=42)
        task_ids = kmeans.fit_predict(self.features)
        clustered_tasks = {i: [] for i in range(num_tasks)}
        for idx, cluster_id in enumerate(task_ids):
            clustered_tasks[cluster_id].append(self.image_paths[idx])
        return clustered_tasks

    # Spectral Clustering
    def cosine_cluster_tasks(self, num_tasks=5, metric='cosine'):
        similarity = self.compute_similarity(metric)
        assert similarity.shape[0] == similarity.shape[1]
        print(similarity.shape)
        clustering = SpectralClustering(n_clusters=num_tasks, affinity='precomputed', random_state=42)
        task_ids = clustering.fit_predict(similarity)
        clustered_tasks = {i: [] for i in range(num_tasks)}
        for idx, cluster_id in enumerate(task_ids):
            clustered_tasks[cluster_id].append(self.image_paths[idx])
        return clustered_tasks

    # GMM clustering
    def gmm_clustering(self):
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(self.features)
        task_ids = gmm.fit_predict(self.features)
        clustered_tasks = {i: [] for i in range(5)}
        for idx, cluster_id in enumerate(task_ids):
            clustered_tasks[cluster_id].append(self.image_paths[idx])
        return clustered_tasks

    #Task sampling
    def sample_task_images(self, task_ids, k_support=2, k_query=1):
        if len(task_ids) < k_support + k_query:
            raise ValueError(f"Not enough images in task_ids. Required: {k_support + k_query}, got: {len(task_ids)}")
        sampled = random.sample(task_ids, k_support + k_query)
        support_paths = sampled[:k_support]
        query_paths = sampled[k_support:]

        def get_index(path):
            filename = os.path.basename(path)
            return self.dataset.image_filenames.index(filename)

        support_ids = [get_index(p) for p in support_paths]
        query_ids = [get_index(p) for p in query_paths]

        return support_ids, query_ids