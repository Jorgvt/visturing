import os
import re
from glob import glob
from collections import defaultdict

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as stats

from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth, calculate_correlations, prepare_and_correlate, prepare_and_correlate_order, calculate_spearman, calculate_pearson_stack

def load_ground_truth(root_path: str = "../../ground_truth_decalogo", # Path to the root containing all the ground truth files
                      ): # Tuple (x, y, y_rg, y_yb)
    data = sio.loadmat(os.path.join(root_path, "responses_no_mask_achrom_1p5_3_6_12_24.mat"))
    data = data["resp_no_mask_achrom"]
    x, y = data[0], data[1:]
    
    data = sio.loadmat(os.path.join(root_path, "responses_no_mask_RG_1p5_3_6_12_24.mat"))
    data = data["resp_no_mask_RG"]
    x, y_rg = data[0], data[1:]
    
    data = sio.loadmat(os.path.join(root_path, "responses_no_mask_YB_1p5_3_6_12_24.mat"))
    data = data["resp_no_mask_YB"]
    x, y_yb = data[0], data[1:]
    
    return x, y, y_rg, y_yb

def load_data(root_path):
    c_a = np.load(os.path.join(root_path, "contrast_a.npy"))
    c_rg = np.load(os.path.join(root_path, "contrast_rg.npy"))
    c_yb = np.load(os.path.join(root_path, "contrast_yb.npy"))
    return c_a, c_rg, c_yb

def plot_ground_truth(x,
                      y,
                      y_rg,
                      y_yb,
                      figsize=(14,4),
                      ): # Returns both the fig and axes objects
    freqs = [1.5, 3, 6, 12, 24]
    colors = ["lightgray", "black", "blue", "gray", "red"]
    colors_rg = ["blue", "black", "gray", "lightgray", "red"]
    colors_yb = ["blue", "black", "gray", "lightgray", "red"]

    fig, axes = plt.subplots(1,3, figsize=figsize, sharex=True, sharey=True)
    for y_, f, c in zip(y, freqs, colors):
        axes[0].plot(x, y_, color=c, label=f"Freq {f}")
    for y_, f, c in zip(y_rg, freqs, colors_rg):
        axes[1].plot(x, y_, color=c, label=f"Freq {f}")
    for y_, f, c in zip(y_yb, freqs, colors_yb):
        axes[2].plot(x, y_, color=c, label=f"Freq {f}")
    axes[0].set_ylabel("Visibility")
    for ax in axes.ravel():
        ax.set_xlabel("Contrast")
        ax.legend()
    return fig, axes
    
def evaluate(calculate_diffs,
             data_path,
             gt_path,
             ):

    x_gt, y_gt, rg_gt, yb_gt = load_ground_truth(gt_path)
    data = {re.findall("noise_(\w+)\.", p)[0]: np.load(p) for p in glob(os.path.join(data_path, "*")) if "gabor" in p}
    freqs = np.array([1.5, 3, 6, 12, 24])

    diffs = defaultdict(dict)
    for name, chroma in data.items():
        for f, dat in zip(freqs, chroma):
            diffs_ = calculate_diffs(dat, dat[0:1])
            diffs[name][f] = diffs_

    x_a, x_rg, x_yb = load_data(data_path)
    
    diffs_a = np.array([a for a in diffs["achrom"].values()])
    diffs_rg = np.array([a for a in diffs["red_green"].values()])
    diffs_yb = np.array([a for a in diffs["yellow_blue"].values()])

    bs, ds = [], []
    for b, d in zip(diffs_a, y_gt):
        a, b, c, d = prepare_data(x_a, b, x_gt, d)
        bs.append(b)
        ds.append(d)
    b_a = np.array(bs)
    d_a = np.array(ds)

    order_corr = {}
    order_corr["achrom"] = calculate_correlations_with_ground_truth(b_a, d_a)

    bs, ds = [], []
    for b, d in zip(diffs_rg, rg_gt):
        a, b, c, d = prepare_data(x_rg, b, x_gt, d)
        bs.append(b)
        ds.append(d)
    b_rg = np.array(bs)
    d_rg = np.array(ds)


    order_corr["red_green"] = calculate_correlations_with_ground_truth(b_rg, d_rg)


    bs, ds = [], []
    for b, d in zip(diffs_yb, yb_gt):
        a, b, c, d = prepare_data(x_yb, b, x_gt, d)
        bs.append(b)
        ds.append(d)
    b_yb = np.array(bs)
    d_yb = np.array(ds)


    order_corr["yellow_blue"] = calculate_correlations_with_ground_truth(b_yb, d_yb)


    b_cat = np.concatenate([
            b_a.ravel(), b_rg.ravel(), b_yb.ravel(),
        ])
    nan_mask = np.isnan(b_cat)
    d_cat = np.concatenate([
            d_a.ravel(), d_rg.ravel(), d_yb.ravel(),
        ])
    pearson = stats.pearsonr(
        b_cat[~nan_mask], d_cat[~nan_mask]
    )

    return order_corr, pearson
