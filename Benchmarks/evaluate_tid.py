#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "torchvision",
#   "numpy",
#   "pandas",
#   "scipy",
#   "datasets",
#   "huggingface-hub",
#   "tabulate",
# ]
# ///

import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from scipy.stats import pearsonr, spearmanr, kendalltau
from datasets import load_dataset

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Add the parent directory to sys.path to import local visturing package if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, repo_root)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

# Define the wrapper class for ResNet-18
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights).to(device)
        except AttributeError:
            self.model = models.resnet18(pretrained=True).to(device)
        self.model.eval()
        self.device = device
        
        # Target layers
        self.feature_layers = {
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4,
            'avgpool': self.model.avgpool,
            'fc': self.model.fc
        }
        
        self.features = {}
        self.hooks = []
        for name, layer in self.feature_layers.items():
            hook = layer.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)
            
    def _make_hook(self, name):
        def hook_fn(module, input, output):
            self.features[name] = output
        return hook_fn
        
    def forward(self, x):
        self.features.clear()
        with torch.no_grad():
            _ = self.model(x)
        return self.features
        
    def close(self):
        for hook in self.hooks:
            hook.remove()

# Initialize feature extractor
extractor = ResNet18FeatureExtractor(device)

def evaluate_on_dataset(ds, layer_name, batch_size=32):
    """Processes the dataset in batches, extracting features and computing distances."""
    distances = []
    mos_values = []
    
    # ImageNet normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    num_samples = len(ds)
    for i in range(0, num_samples, batch_size):
        batch = ds[i : i + batch_size]
        
        dist_imgs = []
        ref_imgs = []
        
        for dist_img, ref_img in zip(batch['distorted'], batch['reference']):
            # Convert to numpy and normalize to [0, 1]
            dist_arr = np.array(dist_img.convert('RGB')) / 255.0
            ref_arr = np.array(ref_img.convert('RGB')) / 255.0
            dist_imgs.append(dist_arr)
            ref_imgs.append(ref_arr)
            
        dist_batch = np.stack(dist_imgs, axis=0)
        ref_batch = np.stack(ref_imgs, axis=0)
        
        # Convert to torch tensor (B, H, W, C) -> (B, C, H, W)
        x_dist = torch.from_numpy(dist_batch).float().permute(0, 3, 1, 2).to(device)
        x_ref = torch.from_numpy(ref_batch).float().permute(0, 3, 1, 2).to(device)
        
        # Normalize
        x_dist_norm = (x_dist - mean) / std
        x_ref_norm = (x_ref - mean) / std
        
        # Forward pass
        with torch.no_grad():
            feats_dist = extractor(x_dist_norm)
            feat_dist = feats_dist[layer_name]
            
            feats_ref = extractor(x_ref_norm)
            feat_ref = feats_ref[layer_name]
            
            # Compute RMSE between features (flat RMSE)
            diff = feat_dist - feat_ref
            diff_flat = diff.reshape(diff.size(0), -1)
            batch_dists = torch.sqrt(torch.mean(diff_flat ** 2, dim=1))
            
            distances.extend(batch_dists.cpu().numpy().tolist())
            
        mos_values.extend(batch['mos'])
        
    distances = np.array(distances)
    mos_values = np.array(mos_values)
    
    # Calculate correlations
    plcc, _ = pearsonr(distances, mos_values)
    srocc, _ = spearmanr(distances, mos_values)
    krocc, _ = kendalltau(distances, mos_values)
    
    return {
        "PLCC": plcc,
        "SROCC": srocc,
        "KROCC": krocc,
        "|PLCC|": abs(plcc),
        "|SROCC|": abs(srocc),
        "|KROCC|": abs(krocc)
    }

# Load datasets from Hugging Face
print("Loading TID2008 dataset...")
tid2008 = load_dataset("Jorgvt/TID2008", split="train")
print(f"Loaded TID2008 with {len(tid2008)} samples.")

print("Loading TID2013 dataset...")
tid2013 = load_dataset("Jorgvt/TID2013", split="train")
print(f"Loaded TID2013 with {len(tid2013)} samples.")

layers = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Dictionary to hold summarized rows
tid2008_rows = []
tid2013_rows = []

for layer in layers:
    print(f"\nEvaluating ResNet-18 layer: {layer} on TID2008...")
    res_08 = evaluate_on_dataset(tid2008, layer, batch_size=32)
    res_08["Layer"] = layer
    tid2008_rows.append(res_08)
    
    print(f"Evaluating ResNet-18 layer: {layer} on TID2013...")
    res_13 = evaluate_on_dataset(tid2013, layer, batch_size=32)
    res_13["Layer"] = layer
    tid2013_rows.append(res_13)

# Build dataframes
df_2008 = pd.DataFrame(tid2008_rows)[["Layer", "PLCC", "SROCC", "KROCC", "|PLCC|", "|SROCC|", "|KROCC|"]]
df_2013 = pd.DataFrame(tid2013_rows)[["Layer", "PLCC", "SROCC", "KROCC", "|PLCC|", "|SROCC|", "|KROCC|"]]

# Save to CSV
tid2008_csv = os.path.join(results_dir, "resnet18_tid2008_correlations.csv")
tid2013_csv = os.path.join(results_dir, "resnet18_tid2013_correlations.csv")

df_2008.to_csv(tid2008_csv, index=False)
df_2013.to_csv(tid2013_csv, index=False)

print("\n=========================================")
print("TID2008 Results Summary:")
print(df_2008.to_markdown(index=False))
print("\n=========================================")
print("TID2013 Results Summary:")
print(df_2013.to_markdown(index=False))
print("=========================================")

# Clean up hooks
extractor.close()

# Save visual summary Markdown
summary_md_path = os.path.join(results_dir, "tid_summary.md")
with open(summary_md_path, "w") as f:
    f.write("# ResNet-18 TID2008 & TID2013 Evaluation Summary\n\n")
    f.write("This document summarizes the performance of a pretrained ResNet-18 model evaluated at multiple layers on the TID2008 and TID2013 image quality assessment benchmarks.\n\n")
    
    f.write("## TID2008 Evaluation Results\n\n")
    f.write("The table below reports Pearson (PLCC), Spearman (SROCC), and Kendall (KROCC) correlation coefficients between feature-based model distances and subjective Mean Opinion Scores (MOS).\n\n")
    f.write(df_2008.to_markdown(index=False) + "\n\n")
    
    f.write("## TID2013 Evaluation Results\n\n")
    f.write("The table below reports the same metrics on the larger TID2013 benchmark dataset.\n\n")
    f.write(df_2013.to_markdown(index=False) + "\n\n")
    
    f.write("> [!NOTE]\n")
    f.write("> Since model distance is inversely related to perceived image quality (MOS), the raw correlation values (PLCC, SROCC, KROCC) are negative. The absolute values ($|\\text{PLCC}|$, $|\\text{SROCC}|$, $|\\text{KROCC}|$) represent the strength of the alignment.\n\n")

print(f"Saved TID summary Markdown to: {summary_md_path}")
