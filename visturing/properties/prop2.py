import os
from glob import glob
import wget
from zipfile import ZipFile

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from .math_utils import pearson_correlation

from visturing.ranking import prepare_data, calculate_spearman

def load_ground_truth(data_path: str = "ground_truth_decalogo", # Path to the root containing all the ground truth files
                      ): # Tuple of tuples (x, y), (x_c, red-green), (x_c, yellow-blue)
    data = sio.loadmat(os.path.join(data_path, "weber.mat"))
    data = data["weber"]

    x = data[0]
    y = data[1]

    data = sio.loadmat(os.path.join(data_path, "resp_RG.mat"))
    data = data["resp_RG"]
    x_c, rg, _ = data

    data = sio.loadmat(os.path.join(data_path, "resp_YB.mat"))
    data = data["resp_YB"]
    _, yb = data
    
    return x, y, x_c, rg, x_c, yb

def load_data(data_path):
    x_a = np.load(os.path.join(data_path, "luminancias.npy"))
    x_rg = np.load(os.path.join(data_path, "x_rg.npy"))
    x_yb = np.load(os.path.join(data_path, "x_yb.npy"))
    return x_a, x_rg, x_yb

def load_images(data_path):
    data = {p.split("/")[-1].split(".")[0]: np.load(p, allow_pickle=True) for p in glob(os.path.join(data_path, "*npy")) if "bgs" not in p and "luminancias" not in p and "x_rg" not in p and "x_yb" not in p}
    bgs = {p.split("/")[-1].split(".")[0][4:]: np.load(p, allow_pickle=True) for p in glob(os.path.join(data_path, "*npy")) if "bgs" in p}
    return data, bgs


def plot_ground_truth(x,y,
                      x_c, rg,
                      x_cc, yb,
                      ): # Returns both the fig and axes objects
    fig, axes = plt.subplots(1,3, figsize=(18,5))
    axes[0].plot(x, y, "b")
    axes[1].plot(x_c, rg, "gray")
    axes[2].plot(x_c, yb, "gray")
    for ax, name in zip(axes, ["Achromatic", "Red-Green", "Yellow-Blue"]): ax.set_title(name)
    axes[0].set_xlabel(r"Luminance (cd/m$^2$)")
    axes[1].set_xlabel("Linear RG")
    axes[2].set_xlabel("Linear YB")
    for ax, name in zip(axes, ["Brightness", "Nonlinear RG", "Nonlinear YB"]): ax.set_ylabel(name)
    axes[1].set_xlim([-22,22])
    axes[1].set_ylim([-8,8])
    axes[2].set_xlim([-22,22])
    axes[2].set_ylim([-8,8])
    return fig, axes

def evaluate(calculate_diffs,
             data_path: str = "Data/Experiment_2",
             gt_path: str = "ground_truth_decalogo",
             ): # Tuple (responses, correlations)

    if not os.path.exists(data_path):
        data_path = download_data("/".join(data_path.split("/")[:-1]))

    ## Load ground truth
    x_a_gt, y_a_gt, x_rg_gt, y_rg_gt, x_yb_gt, y_yb_gt = load_ground_truth(gt_path)

    ## Load data
    x_a, x_rg, x_yb = load_data(data_path)

    data = {p.split("/")[-1].split(".")[0]: np.load(p) for p in glob(os.path.join(data_path, "*npy")) if "bgs" not in p}
    bgs = {p.split("/")[-1].split(".")[0][4:]: np.load(p) for p in glob(os.path.join(data_path, "*npy")) if "bgs" in p}

    diffs = {}
    for c in ["achrom", "red_green", "yellow_blue"]:
        diffs[c] = []
        data_ = data[c]
        bgs_ = bgs[c]
        for cc, bg in zip(data_, bgs_):
            diff = calculate_diffs(cc, bg[None,...])
            diffs[c].append(diff)
    diffs = {k: np.array(v) for k, v in diffs.items()}

    ## Calculate Order Correlation
    spearman_correlations = {}

    a, b, c, d = prepare_data(x_a, diffs["achrom"], x_a_gt, y_a_gt)
    spearman_correlations["achrom"] = calculate_spearman(b, ideal_ordering=[0,1,2,3,4])

    a, b_rg, c, d_rg = prepare_data(x_rg, diffs["red_green"], x_rg_gt, y_rg_gt)
    spearman_correlations["red_green"] = calculate_spearman(b_rg, ideal_ordering=[0,1,2,3,4])

    a, b_yb, c, d_yb = prepare_data(x_yb, diffs["yellow_blue"], x_yb_gt, y_yb_gt)
    spearman_correlations["yellow_blue"] = calculate_spearman(b_yb, ideal_ordering=[0,1,2,3,4])

    # Calculate Pearson
    ## Achromatic

    corr_achrom = pearson_correlation(
        np.concatenate([
            b[0].ravel(),
        ]),
        np.concatenate([
            d.ravel(),
        ])
    )
    ## Both Chromatics Together
    corr_chroma = pearson_correlation(
        np.concatenate([
            b_rg[2].ravel(), b_yb[2].ravel(),
        ]),
        np.concatenate([
            d_rg.ravel(), d_yb.ravel(),
        ])
    )
    correlations = {"pearson_achrom": corr_achrom, "pearson_chrom": corr_chroma, "kendall_corr": spearman_correlations}

    return {"diffs": diffs, "correlations": correlations}

def download_data(data_path, # Path to download the data
                  ):
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    data_url = "https://zenodo.org/records/17700252/files/Experiment_2.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(data_path, "Experiment_2")
