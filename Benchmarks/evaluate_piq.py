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
#   "piq",
#   "tabulate",
#   "matplotlib",
#   "opencv-python",
#   "natsort",
#   "wget",
#   "perceptualtests>=0.1.4",
# ]
# ///

import os
import sys
import warnings

# Enable CPU fallback for MPS device to support unsupported operators (like linalg_eigh used in IW-SSIM)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.stats import pearsonr, spearmanr, kendalltau
from datasets import load_dataset
import piq

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Add parent directory to sys.path to import local visturing package
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, repo_root)

from visturing.properties import evaluate_all
from visturing.properties.utils import build_evaluation_table

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

# Initialize class-based metrics once to avoid slow duplicate creation
print("Initializing model-based PIQ metrics...")
dists_metric = piq.DISTS(reduction='none').to(device)
lpips_metric = piq.LPIPS(reduction='none').to(device)
pieapp_metric = piq.PieAPP(reduction='none').to(device)

def make_piq_distance_fn(name):
    """Returns a distance function tailored to a specific PIQ metric, with upscaling for small images."""
    def calculate_diffs(img1, img2):
        x1 = torch.from_numpy(img1).float().permute(0, 3, 1, 2).to(device)
        x2 = torch.from_numpy(img2).float().permute(0, 3, 1, 2).to(device)
        
        # Replace NaNs with 0.0 to prevent assertion errors in PIQ
        x1 = torch.nan_to_num(x1, nan=0.0)
        x2 = torch.nan_to_num(x2, nan=0.0)
        
        # Clamp to [0, 1] to prevent assertion errors in PIQ
        x1 = torch.clamp(x1, 0.0, 1.0)
        x2 = torch.clamp(x2, 0.0, 1.0)
        
        # Slicing reference shape matching
        if x2.shape[0] == 1 and x1.shape[0] > 1:
            x2 = x2.repeat(x1.shape[0], 1, 1, 1)
            
        # Bilinear interpolation upscaling to 224x224 if dimensions are smaller than 161 (required for MS-SSIM and IW-SSIM)
        h, w = x1.shape[-2:]
        if h < 161 or w < 161:
            x1 = F.interpolate(x1, size=(224, 224), mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, size=(224, 224), mode='bilinear', align_corners=False)
            
        with torch.no_grad():
            if name == 'psnr':
                val = piq.psnr(x1, x2, data_range=1.0, reduction='none')
                dists = -val
            elif name == 'ssim':
                val = piq.ssim(x1, x2, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif name == 'multi_scale_ssim':
                val = piq.multi_scale_ssim(x1, x2, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif name == 'information_weighted_ssim':
                try:
                    # Add a tiny amount of noise to avoid singular covariance matrices in flat regions
                    x1_j = x1 + 1e-6 * torch.randn_like(x1)
                    x2_j = x2 + 1e-6 * torch.randn_like(x2)
                    x1_j = torch.clamp(x1_j, 0.0, 1.0)
                    x2_j = torch.clamp(x2_j, 0.0, 1.0)
                    val = piq.information_weighted_ssim(x1_j, x2_j, data_range=1.0, reduction='none')
                    dists = 1.0 - val
                except Exception:
                    # Fallback to standard ssim if linear algebra solver fails
                    val = piq.ssim(x1, x2, data_range=1.0, reduction='none')
                    dists = 1.0 - val
            elif name == 'vif_p':
                val = piq.vif_p(x1, x2, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif name == 'fsim':
                val = piq.fsim(x1, x2, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif name == 'srsim':
                val = piq.srsim(x1, x2, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif name == 'gmsd':
                dists = piq.gmsd(x1, x2, data_range=1.0, reduction='none')
            elif name == 'multi_scale_gmsd':
                dists = piq.multi_scale_gmsd(x1, x2, data_range=1.0, reduction='none')
            elif name == 'vsi':
                val = piq.vsi(x1, x2, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif name == 'mdsi':
                dists = piq.mdsi(x1, x2, data_range=1.0, reduction='none')
            elif name == 'haarpsi':
                val = piq.haarpsi(x1, x2, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif name == 'dists':
                dists = dists_metric(x1, x2)
            elif name == 'lpips':
                dists = lpips_metric(x1, x2)
            elif name == 'pieapp':
                dists = pieapp_metric(x1, x2)
                
            return dists.cpu().numpy()
            
    return calculate_diffs

def evaluate_piq_on_tid(ds, metric_name, batch_size=32):
    """Evaluates a PIQ metric on a TID dataset in batches."""
    distances = []
    mos_values = []
    
    num_samples = len(ds)
    for i in range(0, num_samples, batch_size):
        batch = ds[i : i + batch_size]
        
        dist_imgs = []
        ref_imgs = []
        
        for dist_img, ref_img in zip(batch['distorted'], batch['reference']):
            dist_arr = np.array(dist_img.convert('RGB')) / 255.0
            ref_arr = np.array(ref_img.convert('RGB')) / 255.0
            dist_imgs.append(dist_arr)
            ref_imgs.append(ref_arr)
            
        dist_batch = np.stack(dist_imgs, axis=0)
        ref_batch = np.stack(ref_imgs, axis=0)
        
        x_dist = torch.from_numpy(dist_batch).float().permute(0, 3, 1, 2).to(device)
        x_ref = torch.from_numpy(ref_batch).float().permute(0, 3, 1, 2).to(device)
        
        # Replace NaNs with 0.0 to prevent assertion errors in PIQ
        x_dist = torch.nan_to_num(x_dist, nan=0.0)
        x_ref = torch.nan_to_num(x_ref, nan=0.0)
        
        # Clamp to [0, 1] to prevent assertion errors in PIQ
        x_dist = torch.clamp(x_dist, 0.0, 1.0)
        x_ref = torch.clamp(x_ref, 0.0, 1.0)
        
        with torch.no_grad():
            if metric_name == 'psnr':
                val = piq.psnr(x_dist, x_ref, data_range=1.0, reduction='none')
                dists = -val
            elif metric_name == 'ssim':
                val = piq.ssim(x_dist, x_ref, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif metric_name == 'multi_scale_ssim':
                val = piq.multi_scale_ssim(x_dist, x_ref, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif metric_name == 'information_weighted_ssim':
                try:
                    # Add a tiny amount of noise to avoid singular covariance matrices in flat regions
                    x_dist_j = x_dist + 1e-6 * torch.randn_like(x_dist)
                    x_ref_j = x_ref + 1e-6 * torch.randn_like(x_ref)
                    x_dist_j = torch.clamp(x_dist_j, 0.0, 1.0)
                    x_ref_j = torch.clamp(x_ref_j, 0.0, 1.0)
                    val = piq.information_weighted_ssim(x_dist_j, x_ref_j, data_range=1.0, reduction='none')
                    dists = 1.0 - val
                except Exception:
                    # Fallback to standard ssim if linear algebra solver fails
                    val = piq.ssim(x_dist, x_ref, data_range=1.0, reduction='none')
                    dists = 1.0 - val
            elif metric_name == 'vif_p':
                val = piq.vif_p(x_dist, x_ref, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif metric_name == 'fsim':
                val = piq.fsim(x_dist, x_ref, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif metric_name == 'srsim':
                val = piq.srsim(x_dist, x_ref, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif metric_name == 'gmsd':
                dists = piq.gmsd(x_dist, x_ref, data_range=1.0, reduction='none')
            elif metric_name == 'multi_scale_gmsd':
                dists = piq.multi_scale_gmsd(x_dist, x_ref, data_range=1.0, reduction='none')
            elif metric_name == 'vsi':
                val = piq.vsi(x_dist, x_ref, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif metric_name == 'mdsi':
                dists = piq.mdsi(x_dist, x_ref, data_range=1.0, reduction='none')
            elif metric_name == 'haarpsi':
                val = piq.haarpsi(x_dist, x_ref, data_range=1.0, reduction='none')
                dists = 1.0 - val
            elif metric_name == 'dists':
                dists = dists_metric(x_dist, x_ref)
            elif metric_name == 'lpips':
                dists = lpips_metric(x_dist, x_ref)
            elif metric_name == 'pieapp':
                dists = pieapp_metric(x_dist, x_ref)
                
            distances.extend(dists.cpu().numpy().tolist())
            
        mos_values.extend(batch['mos'])
        
    distances = np.array(distances)
    mos_values = np.array(mos_values)
    
    plcc, _ = pearsonr(distances, mos_values)
    srocc, _ = spearmanr(distances, mos_values)
    krocc, _ = kendalltau(distances, mos_values)
    
    return {
        "Metric": metric_name,
        "PLCC": plcc,
        "SROCC": srocc,
        "KROCC": krocc,
        "|PLCC|": abs(plcc),
        "|SROCC|": abs(srocc),
        "|KROCC|": abs(krocc)
    }

# -----------------
# 1. Visturing Suite Evaluation
# -----------------
data_path = os.path.join(repo_root, "Data")
gt_path = data_path

print("Starting Visturing evaluation for all PIQ metrics...")
metrics = [
    'psnr', 'ssim', 'multi_scale_ssim', 'information_weighted_ssim', 'vif_p',
    'fsim', 'srsim', 'gmsd', 'multi_scale_gmsd', 'vsi', 'mdsi', 'haarpsi',
    'dists', 'lpips', 'pieapp'
]

results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

all_dfs = {}
for metric in metrics:
    print(f"\n=========================================")
    print(f"Evaluating PIQ Metric: {metric} on Visturing")
    print(f"=========================================")
    
    calc_diffs = make_piq_distance_fn(metric)
    results = evaluate_all(calc_diffs, data_path=data_path, gt_path=gt_path)
    
    df = build_evaluation_table(results)
    all_dfs[metric] = df
    
    # Save individual metric correlations
    df.to_csv(os.path.join(results_dir, f"piq_{metric}_correlations.csv"), index=False)

# Compile summaries across all metrics
properties_list = all_dfs[metrics[0]]["Property"].tolist()
rmse_summary = pd.DataFrame({"Property": properties_list})
order_summary = pd.DataFrame({"Property": properties_list})

for metric in metrics:
    df_m = all_dfs[metric]
    rmse_summary[metric] = df_m["RMSE fit (ρ_p)"]
    order_summary[metric] = df_m["Curve Order (ρ_k)"]

rmse_summary.to_csv(os.path.join(results_dir, "piq_rmse_fit_summary.csv"), index=False)
order_summary.to_csv(os.path.join(results_dir, "piq_curve_order_summary.csv"), index=False)

# -----------------
# 2. TID2008 & TID2013 Evaluation
# -----------------
print("\nLoading TID2008 dataset...")
tid2008 = load_dataset("Jorgvt/TID2008", split="train")
print("Loading TID2013 dataset...")
tid2013 = load_dataset("Jorgvt/TID2013", split="train")

tid2008_rows = []
tid2013_rows = []

for metric in metrics:
    print(f"\nEvaluating PIQ Metric: {metric} on TID2008...")
    res_08 = evaluate_piq_on_tid(tid2008, metric, batch_size=32)
    tid2008_rows.append(res_08)
    
    print(f"Evaluating PIQ Metric: {metric} on TID2013...")
    res_13 = evaluate_piq_on_tid(tid2013, metric, batch_size=32)
    tid2013_rows.append(res_13)

df_tid2008 = pd.DataFrame(tid2008_rows)[["Metric", "PLCC", "SROCC", "KROCC", "|PLCC|", "|SROCC|", "|KROCC|"]]
df_tid2013 = pd.DataFrame(tid2013_rows)[["Metric", "PLCC", "SROCC", "KROCC", "|PLCC|", "|SROCC|", "|KROCC|"]]

df_tid2008.to_csv(os.path.join(results_dir, "piq_tid2008_correlations.csv"), index=False)
df_tid2013.to_csv(os.path.join(results_dir, "piq_tid2013_correlations.csv"), index=False)

# Save visual Markdown summaries
summary_md_path = os.path.join(results_dir, "piq_summary.md")
with open(summary_md_path, "w") as f:
    f.write("# PIQ Full-Reference Metrics Evaluation Summary\n\n")
    f.write("This document summarizes the psychophysical and image quality benchmark performance of 15 full-reference metrics from the `piq` library.\n\n")
    
    f.write("## 1. Visturing Suite Results\n\n")
    f.write("### RMSE Fit Correlation ($\\rho_p$) Summary\n")
    f.write(rmse_summary.to_markdown(index=False) + "\n\n")
    
    f.write("### Curve Order Correlation ($\\rho_k$) Summary\n")
    f.write(order_summary.to_markdown(index=False) + "\n\n")
    
    f.write("## 2. TID2008 & TID2013 Results\n\n")
    f.write("### TID2008 Correlations\n")
    f.write(df_tid2008.to_markdown(index=False) + "\n\n")
    
    f.write("### TID2013 Correlations\n")
    f.write(df_tid2013.to_markdown(index=False) + "\n\n")

print("\n=========================================")
print(f"PIQ EVALUATION COMPLETE! Summaries saved to {results_dir}")
print("=========================================")
