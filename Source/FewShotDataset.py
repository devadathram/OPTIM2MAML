from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import cv2
import os
import random
from collections import defaultdict

#Dataset loading with preprocessing
class PNGSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        #self.mask_filenames = [f.replace('.png', '_mask.png') for f in self.image_filenames]
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #CLAHE transformation
        clahe_img = clahe.apply(image)
        clahe_img = (clahe_img - np.mean(clahe_img)) / (np.std(clahe_img) + 1e-8) #normalization
        clahe_img = torch.tensor(clahe_img, dtype=torch.float32).unsqueeze(0) / 255.0
        mask = Image.open(mask_path).convert("L")  # Convert to 1-channel grayscale

        if self.transform:
            image = self.transform(clahe_img)

        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()  # Binarize

        return image, mask

#Batch of tasks creation
class SegmentationTaskDataset:
    def __init__(self, dataset, k_shot=5, q_query=1, episodes=1000):
        self.dataset = dataset
        self.k = k_shot
        self.q = q_query
        self.episodes = episodes

        self.patient_map = defaultdict(list) #task creation using patient_ids
        for idx, fname in enumerate(dataset.image_filenames):
            patient_id = fname.split('_')[0]
            self.patient_map[patient_id].append(idx)

        self.patients = list(self.patient_map.keys())
        print(self.patients)

    def __len__(self):
        return self.episodes

    def __iter__(self):
        for _ in range(self.episodes):
            patient = random.choice(self.patients)
            indices = self.patient_map[patient]

            if len(indices) < self.k + self.q:
                continue  # skip if not enough slices

            selected = random.sample(indices, self.k + self.q)
            support_idxs = selected[:self.k]
            query_idxs = selected[self.k:]

            yield support_idxs, query_idxs

