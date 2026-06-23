#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers",
#   "numpy",
#   "pandas",
#   "scipy",
#   "datasets",
#   "huggingface-hub",
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
import argparse
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.stats import pearsonr, spearmanr, kendalltau
from datasets import load_dataset
from transformers import CLIPVisionModel

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

# Global hook variables
features = {}
hooks = []

def register_hooks(feature_layers):
    """Registers hooks on the target modules to capture intermediate outputs."""
    global hooks
    # Remove any existing hooks
    for h in hooks:
        h.remove()
    hooks.clear()
    features.clear()
    
    def make_hook(name):
        def hook_fn(module, input, output):
            features[name] = output
        return hook_fn
        
    for name, layer in feature_layers.items():
        h = layer.register_forward_hook(make_hook(name))
        hooks.append(h)

def clean_hooks():
    """Removes all registered hooks."""
    global hooks
    for h in hooks:
        h.remove()
    hooks.clear()
    features.clear()

def extract_representation(name, output):
    """Extracts a 2D or 4D feature representation. Extracts CLS token for 3D transformer tokens."""
    if isinstance(output, tuple):
        output = output[0]
    if len(output.shape) == 3:
        # Transformer sequence: shape (B, S, D). Extract CLS token at index 0.
        return output[:, 0, :]
    return output

def main():
    parser = argparse.ArgumentParser(description="Evaluate vision models on Visturing and TID suites.")
    parser.add_argument("-m", "--model", type=str, default="resnet18",
                        choices=["resnet18", "vgg16", "convnext", "vit", "clip", "dinov2"],
                        help="Vision model to evaluate.")
    parser.add_argument("-d", "--dataset", type=str, default="all",
                        choices=["visturing", "tid", "all"],
                        help="Dataset to evaluate on.")
    parser.add_argument("-b", "--batch-size", type=str, default="32",
                        help="Batch size for evaluation.")
    args = parser.parse_args()
    
    batch_size = int(args.batch_size)
    model_name = args.model
    dataset_name = args.dataset
    
    print(f"Using device: {device}")
    print(f"Loading model: {model_name}...")
    
    # 1. Model Loading & Hook registration
    if model_name == "resnet18":
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
        except AttributeError:
            model = models.resnet18(pretrained=True).to(device)
        model.eval()
        
        feature_layers = {
            'layer1': model.layer1,
            'layer2': model.layer2,
            'layer3': model.layer3,
            'layer4': model.layer4,
            'avgpool': model.avgpool,
            'fc': model.fc
        }
        
    elif model_name == "vgg16":
        try:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
        except AttributeError:
            model = models.vgg16(pretrained=True).to(device)
        model.eval()
        
        feature_layers = {
            'conv2': model.features[9],  # After second block
            'conv3': model.features[16], # After third block
            'conv4': model.features[23], # After fourth block
            'conv5': model.features[30], # After fifth block
            'avgpool': model.avgpool,
            'fc': model.classifier[6]
        }
        
    elif model_name == "convnext":
        try:
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT).to(device)
        except AttributeError:
            model = models.convnext_tiny(pretrained=True).to(device)
        model.eval()
        
        feature_layers = {
            'stage1': model.features[1],
            'stage2': model.features[3],
            'stage3': model.features[5],
            'stage4': model.features[7],
            'avgpool': model.avgpool,
            'fc': model.classifier[2]
        }
        
    elif model_name == "vit":
        try:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device)
        except AttributeError:
            model = models.vit_b_16(pretrained=True).to(device)
        model.eval()
        
        feature_layers = {
            'block0': model.encoder.layers.encoder_layer_0,
            'block3': model.encoder.layers.encoder_layer_3,
            'block7': model.encoder.layers.encoder_layer_7,
            'block11': model.encoder.layers.encoder_layer_11,
            'head': model.heads.head
        }
        
    elif model_name == "clip":
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        model.eval()
        
        feature_layers = {
            'block0': model.vision_model.encoder.layers[0],
            'block3': model.vision_model.encoder.layers[3],
            'block7': model.vision_model.encoder.layers[7],
            'block11': model.vision_model.encoder.layers[11],
            'pooler': model.vision_model.post_layernorm
        }
        
    elif model_name == "dinov2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        model.eval()
        
        feature_layers = {
            'block0': model.blocks[0],
            'block3': model.blocks[3],
            'block7': model.blocks[7],
            'block11': model.blocks[11],
            'norm': model.norm
        }

    register_hooks(feature_layers)
    print(f"Successfully loaded {model_name} and registered hooks.")

    # 2. Define preprocessing norms (CLIP has custom norms, others standard ImageNet)
    if model_name == "clip":
        mean_val = [0.48145466, 0.4578275, 0.40821073]
        std_val = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean_val = [0.485, 0.456, 0.406]
        std_val = [0.229, 0.224, 0.225]
        
    mean = torch.tensor(mean_val, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_val, device=device).view(1, 3, 1, 1)

    # 3. Model feature-distance computation
    def extract_features_batched(imgs, layer_name):
        N = imgs.shape[0]
        features_list = []
        
        is_vit = model_name in ["vit", "clip", "dinov2"]
        
        for i in range(0, N, batch_size):
            batch_imgs = imgs[i : i + batch_size]
            x = torch.from_numpy(batch_imgs).float().permute(0, 3, 1, 2).to(device)
            x_norm = (x - mean) / std
            
            # Upscale/Resize for Transformers
            if is_vit:
                x_norm = F.interpolate(x_norm, size=(224, 224), mode='bilinear', align_corners=False)
                
            features.clear()
            with torch.no_grad():
                _ = model(x_norm)
                
            feat = features[layer_name]
            feat_repr = extract_representation(layer_name, feat)
            features_list.append(feat_repr.cpu())
            
        return torch.cat(features_list, dim=0)

    def make_calculate_diffs(layer_name):
        def calculate_diffs(img1, img2):
            feat1 = extract_features_batched(img1, layer_name)
            feat2 = extract_features_batched(img2, layer_name)
            
            diff = feat1 - feat2
            diff_flat = diff.reshape(diff.size(0), -1)
            dists = torch.sqrt(torch.mean(diff_flat ** 2, dim=1))
            return dists.numpy()
        return calculate_diffs

    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # EVALUATION ON VISTURING
    # ----------------------------------------------------
    if dataset_name in ["visturing", "all"]:
        print(f"\nEvaluating {model_name} on Visturing Suite...")
        data_path = os.path.join(repo_root, "Data")
        gt_path = data_path
        
        layers_results = {}
        for layer in feature_layers.keys():
            print(f"Running layer: {layer}")
            calc_diffs = make_calculate_diffs(layer)
            results = evaluate_all(calc_diffs, data_path=data_path, gt_path=gt_path)
            df = build_evaluation_table(results)
            layers_results[layer] = df
            
            # Save individual CSV
            df.to_csv(os.path.join(results_dir, f"{model_name}_{layer}_correlations.csv"), index=False)
            
        # Compile Visturing summaries
        properties = layers_results[list(feature_layers.keys())[0]]["Property"].tolist()
        rmse_summary = pd.DataFrame({"Property": properties})
        order_summary = pd.DataFrame({"Property": properties})
        
        for layer in feature_layers.keys():
            rmse_summary[layer] = layers_results[layer]["RMSE fit (ρ_p)"]
            order_summary[layer] = layers_results[layer]["Curve Order (ρ_k)"]
            
        rmse_summary.to_csv(os.path.join(results_dir, f"{model_name}_rmse_fit_summary.csv"), index=False)
        order_summary.to_csv(os.path.join(results_dir, f"{model_name}_curve_order_summary.csv"), index=False)
        
        # Save Visturing markdown summary
        with open(os.path.join(results_dir, f"{model_name}_visturing_summary.md"), "w") as f:
            f.write(f"# {model_name.upper()} Visturing Evaluation Summary\n\n")
            f.write("### RMSE Fit Correlation ($\\rho_p$) Summary\n\n")
            f.write(rmse_summary.to_markdown(index=False) + "\n\n")
            f.write("### Curve Order Correlation ($\\rho_k$) Summary\n\n")
            f.write(order_summary.to_markdown(index=False) + "\n\n")
        print(f"Saved Visturing summary to {results_dir}/{model_name}_visturing_summary.md")

    # ----------------------------------------------------
    # EVALUATION ON TID2008 / TID2013
    # ----------------------------------------------------
    if dataset_name in ["tid", "all"]:
        print(f"\nLoading TID datasets for {model_name} evaluation...")
        tid2008 = load_dataset("Jorgvt/TID2008", split="train")
        tid2013 = load_dataset("Jorgvt/TID2013", split="train")
        
        def evaluate_on_tid(ds, layer_name):
            distances = []
            mos_values = []
            num_samples = len(ds)
            
            for i in range(0, num_samples, batch_size):
                batch = ds[i : i + batch_size]
                dist_imgs = []
                ref_imgs = []
                for dist_img, ref_img in zip(batch['distorted'], batch['reference']):
                    dist_imgs.append(np.array(dist_img.convert('RGB')) / 255.0)
                    ref_imgs.append(np.array(ref_img.convert('RGB')) / 255.0)
                    
                dist_batch = np.stack(dist_imgs, axis=0)
                ref_batch = np.stack(ref_imgs, axis=0)
                
                feat_dist = extract_features_batched(dist_batch, layer_name)
                feat_ref = extract_features_batched(ref_batch, layer_name)
                
                diff = feat_dist - feat_ref
                diff_flat = diff.reshape(diff.size(0), -1)
                batch_dists = torch.sqrt(torch.mean(diff_flat ** 2, dim=1))
                
                distances.extend(batch_dists.numpy().tolist())
                mos_values.extend(batch['mos'])
                
            distances = np.array(distances)
            mos_values = np.array(mos_values)
            
            plcc, _ = pearsonr(distances, mos_values)
            srocc, _ = spearmanr(distances, mos_values)
            krocc, _ = kendalltau(distances, mos_values)
            
            return {
                "Layer": layer_name,
                "PLCC": plcc,
                "SROCC": srocc,
                "KROCC": krocc,
                "|PLCC|": abs(plcc),
                "|SROCC|": abs(srocc),
                "|KROCC|": abs(krocc)
            }
            
        tid2008_rows = []
        tid2013_rows = []
        for layer in feature_layers.keys():
            print(f"Evaluating layer {layer} on TID...")
            tid2008_rows.append(evaluate_on_tid(tid2008, layer))
            tid2013_rows.append(evaluate_on_tid(tid2013, layer))
            
        df_tid08 = pd.DataFrame(tid2008_rows)[["Layer", "PLCC", "SROCC", "KROCC", "|PLCC|", "|SROCC|", "|KROCC|"]]
        df_tid13 = pd.DataFrame(tid2013_rows)[["Layer", "PLCC", "SROCC", "KROCC", "|PLCC|", "|SROCC|", "|KROCC|"]]
        
        df_tid08.to_csv(os.path.join(results_dir, f"{model_name}_tid2008_correlations.csv"), index=False)
        df_tid13.to_csv(os.path.join(results_dir, f"{model_name}_tid2013_correlations.csv"), index=False)
        
        # Save TID markdown summary
        with open(os.path.join(results_dir, f"{model_name}_tid_summary.md"), "w") as f:
            f.write(f"# {model_name.upper()} TID Evaluation Summary\n\n")
            f.write("### TID2008 Results Summary\n\n")
            f.write(df_tid08.to_markdown(index=False) + "\n\n")
            f.write("### TID2013 Results Summary\n\n")
            f.write(df_tid13.to_markdown(index=False) + "\n\n")
        print(f"Saved TID summary to {results_dir}/{model_name}_tid_summary.md")

    clean_hooks()
    print(f"\n=========================================")
    print(f"EVALUATION COMPLETE FOR MODEL: {model_name}")
    print(f"Results stored under results/ directory.")
    print(f"=========================================")

if __name__ == "__main__":
    main()
