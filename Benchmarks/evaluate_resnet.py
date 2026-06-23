#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "torchvision",
#   "numpy",
#   "pandas",
#   "scipy",
#   "opencv-python",
#   "natsort",
#   "wget",
#   "perceptualtests>=0.1.4",
#   "tabulate",
#   "matplotlib",
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

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Add the parent directory to sys.path to import local visturing package
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, repo_root)

from visturing.properties import evaluate_all
from visturing.properties.utils import build_evaluation_table

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

# Define the wrapper class for ResNet-18
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load pretrained ResNet-18
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

def extract_features_batched(imgs, layer_name, batch_size=32):
    """Extracts features for a batch of images from the given layer in a memory-safe manner."""
    N = imgs.shape[0]
    features_list = []
    
    # ImageNet normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    for i in range(0, N, batch_size):
        batch_imgs = imgs[i : i + batch_size]
        # Convert to tensor and permute (B, H, W, C) -> (B, C, H, W)
        x = torch.from_numpy(batch_imgs).float().permute(0, 3, 1, 2).to(device)
        # Normalize
        x_norm = (x - mean) / std
        
        # Forward pass to populate hooks
        feats = extractor(x_norm)
        feat = feats[layer_name]
        
        # Store features on CPU
        features_list.append(feat.cpu())
        
    return torch.cat(features_list, dim=0)

def make_calculate_diffs(layer_name):
    """Returns a distance function tailored to a specific ResNet layer."""
    def calculate_diffs(img1, img2):
        # Extract features for both batches
        feat1 = extract_features_batched(img1, layer_name)
        feat2 = extract_features_batched(img2, layer_name)
        
        # Compute RMSE between features (with broadcasting support)
        diff = feat1 - feat2
        diff_flat = diff.reshape(diff.size(0), -1)
        dists = torch.sqrt(torch.mean(diff_flat ** 2, dim=1))
        
        return dists.numpy()
    return calculate_diffs

# Paths
data_path = os.path.join(repo_root, "Data")
gt_path = data_path

print(f"Data directory: {data_path}")
print(f"Ground Truth directory: {gt_path}")

# Run evaluations for each layer
layers = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
all_dfs = {}

results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

for layer in layers:
    print(f"\n=========================================")
    print(f"Evaluating ResNet-18 layer: {layer}")
    print(f"=========================================")
    
    calc_diffs = make_calculate_diffs(layer)
    results = evaluate_all(calc_diffs, data_path=data_path, gt_path=gt_path)
    
    df = build_evaluation_table(results)
    all_dfs[layer] = df
    
    # Save individual layer results to CSV
    layer_csv = os.path.join(results_dir, f"resnet18_{layer}_correlations.csv")
    df.to_csv(layer_csv, index=False)
    print(f"Saved results for {layer} to {layer_csv}")
    
    # Print the table to console
    print(df.to_markdown(index=False))

# Now compile summary tables across all layers
properties_list = all_dfs[layers[0]]["Property"].tolist()

rmse_summary = pd.DataFrame({"Property": properties_list})
order_summary = pd.DataFrame({"Property": properties_list})

for layer in layers:
    df_layer = all_dfs[layer]
    rmse_summary[layer] = df_layer["RMSE fit (ρ_p)"]
    order_summary[layer] = df_layer["Curve Order (ρ_k)"]

# Save combined summaries to CSV
rmse_summary_csv = os.path.join(results_dir, "resnet18_rmse_fit_summary.csv")
order_summary_csv = os.path.join(results_dir, "resnet18_curve_order_summary.csv")

rmse_summary.to_csv(rmse_summary_csv, index=False)
order_summary.to_csv(order_summary_csv, index=False)

print("\n=========================================")
print("EVALUATION COMPLETE!")
print(f"All individual results and summaries stored in: {results_dir}")
print("=========================================")

# Clean up hooks
extractor.close()

# Generate Markdown Summary File
summary_md_path = os.path.join(results_dir, "resnet18_summary.md")

with open(summary_md_path, "w") as f:
    f.write("# ResNet-18 Visturing Suite Evaluation Summary\n\n")
    f.write("This directory contains the psychophysical evaluation results of a pre-trained ResNet-18 model across multiple layers.\n\n")
    f.write("## Overview of Summary Tables\n\n")
    f.write("Below are the comparative tables for both the RMSE fit correlation ($\\rho_p$) and Curve Order correlation ($\\rho_k$) across all evaluated layers.\n\n")
    
    f.write("### 1. RMSE Fit Correlation ($\\rho_p$) Summary\n")
    f.write("This table shows the Pearson correlation between model distances and human psychophysical response functions.\n\n")
    f.write(rmse_summary.to_markdown(index=False) + "\n\n")
    
    f.write("### 2. Curve Order Correlation ($\\rho_k$) Summary\n")
    f.write("This table shows the Spearman/Kendall-tau rank correlation for the ordering of the psychophysical functions.\n\n")
    f.write(order_summary.to_markdown(index=False) + "\n\n")
    
    f.write("## Layer-wise Detailed Tables\n\n")
    for layer in layers:
        f.write(f"### ResNet-18 Layer: `{layer}`\n\n")
        f.write(all_dfs[layer].to_markdown(index=False) + "\n\n")

print(f"Saved visual summary Markdown to: {summary_md_path}")
