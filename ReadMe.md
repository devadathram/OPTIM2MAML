# OPTIM2MAML: A Dual-Optimized Few-Shot Learning framework for Ischemic Stroke Lesion Segmentation in Brain MRI
This repository presents OPTIM2MAML, a dual-optimized few-shot learning framework that integrates Model-Agnostic Meta-Learning (MAML) with a U-Net backbone for ischemic stroke lesion segmentation in brain MRI. The framework employs unsupervised feature clustering to enhance task generation and improve generalization across unseen patient data. Additionally, it incorporates a post-optimization stage using firefly optimization to refine segmentation outcomes.



#### 📬 Contact
[dr7173@srmist.edu.in](mailto:dr7173@srmist.edu.in) • [c16034@srmist.edu.in](mailto:c16034@srmist.edu.in) • [athiram@srmist.edu.in](athiram@srmist.edu.in)


## 📑 Table of Contents
- Abstract
- Workflow
- Dependencies
- Installation
- Model Training
- Model Evaluation

## 🧠 Abstract
 Ischemic stroke is a serious neurological condition caused by reduced blood flow to the brain, leading to oxygen and nutrient deprivation, neuronal cell death and long-term disability. In recent years, Artificial Intelligence (AI)-driven segmentation models have been increasingly employed to assist in the automated identification of stroke lesions from brain MRI scans. However, existing segmentation models face challenges due to limited annotated datasets and difficulty in generalizing across diverse clinical scenarios. To overcome these limitations, few-shot learning techniques have been explored by enabling the models to learn from sparsely labelled instances. Nevertheless, few-shot learning methods encounter issues such as random task sampling and suboptimal segmentation generation that jeopardize both the quality and efficiency of the segmentation process. To address these limitations, we propose a novel dual optimized few-shot learning framework i.e. ‘OPTIM2MAML’ that integrates a pre-optimization module for adaptive task sampling, leveraging spectral clustering with cosine similarity and a post-optimization module that refines the segmentation process with firefly optimization. The proposed OPTIM2MAML framework is analyzed on Brain MRI (ADC and DWI images) of ISLES 2017 and ISLES 2022 datasets to improve the ischemic stroke lesion segmentation.The results confirm the effectiveness of the proposed approach, highlighting its potential as a robust clinical decision-support tool in neuroimaging.

## 🧭 Workflow

![Overall workflow of the proposed OPTIM2MAML framework for segmenting
ischemic stroke lesions](assets/workflow_new.png)
3D MRI volumes from the ISLES 2017 and 2022 datasets are converted into 2D slices with corresponding ground-truth masks. These are preprocessed through resizing (256×256), grayscale conversion, and intensity normalization to [-1, 1].

The OPTIM2MAML framework follows a three-stage pipeline:

- Pre-Optim: Extracts semantic features and clusters training data to create diverse meta-training tasks, promoting generalization.

- MAML: Applies Model-Agnostic Meta-Learning to learn a highly adaptable model initialization.

- Post-Optim: Performs lightweight task-specific optimization to refine segmentation results.

## ⚙️ Dependencies

- Python                 3.9
- learn2learn            0.2.0
- matplotlib             3.9.4
- mealpy                 3.0.1
- nibabel                5.3.2
- numpy                  1.26.3
- opencv-python          4.11.0.86
- pandas                 2.2.3
- scikit-image           0.24.0
- scikit-learn           1.6.1
- torch                  2.7.0+cu118
- torchaudio             2.7.0+cu118
- torchvision            0.22.0+cu118
- ⚠️ Microsoft Visual Studio is required to build C++ extensions for some libraries like learn2learn.

## 🚀 Installation

### Clone the repository
```bash
  git clone https://github.com/devadathram/OPTIM2MAML
```

### Install dependencies
```bash
  pip install -r requirements.txt
```
## 🧪 Model Training

There are three core training scripts provided:

🔹 Traditional U-Net (Supervised Learning)
```bash 
   python Source/TraditionalUnet.py --config configs/traditional_config.yaml
```
- Trains a basic U-Net model using standard supervised learning.


🔹 U-Net + MAML
```bash
   python Source/TraditionalApproach.py --config configs/traditional_config.yaml
```
- Trains U-Net with MAML-based meta-learning. You can adjust few-shot parameters (k_shots, k_query) in the config file.

🔹 OPTIM2MAML (Pre-Optim MAML)
```bash
   python Source/OptimisedApproach.py --config configs/optimised_config.yaml 
```
Trains the meta-learning model using pre-optimized task sampling via clustering.

🔧 Post-optimization using firefly algorithm is implemented separately in FireflyOptimization.py.

## 🧾 Model Evaluation

These scripts are for the evaluation of the trained models on a single test image. Testing scripts are present in the same folder.

🔹Traditional U-Net Evaluation
```bash
   python Source/TraditionalVisualisation.py
```

🔹 OPTIM2MAML Evaluation
```bash
   python Source/OptimisedVisualization.py
```

