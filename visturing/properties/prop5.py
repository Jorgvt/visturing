import os
from glob import glob
import wget
from zipfile import ZipFile

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from .math_utils import pearson_correlation

from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth

def load_ground_truth(root_path: str = "../../ground_truth_decalogo", # Path to the root containing all the ground truth files
                      ): # Tuple (x, y1, y2, y3)
    data = sio.loadmat(os.path.join(root_path, "Campbell_Blakemore.mat"))
    data = data["Campbell_Blakemore"]
    x, y1, y2, y3 = data
    return x, y1, y2, y3

def plot_ground_truth(x,
                      y1,
                      y2,
                      y3,
                      ): # Returns both the fig and axes objects
    fig, axes = plt.subplots()
    axes.plot(x, y1, "k", linestyle="-", label="3 cpd")
    axes.plot(x, y2, "k", linestyle=":", label="6 cpd")
    axes.plot(x, y3, "k", linestyle="-.", label="12 cpd")
    axes.set_xlabel("Frequency (cpd)")
    axes.set_xscale("log")
    axes.set_xlim([1,32])
    axes.legend()
    return fig, axes

def evaluate(calculate_diffs,
             data_path,
             gt_path,
             ):

    if not os.path.exists(data_path):
        data_path = download_data("/".join(data_path.split("/")[:-1]))

    x_gt, y1_gt, y2_gt, y3_gt = load_ground_truth(gt_path)

    noises = {p.split("/")[-1].split(".")[0].split("_")[-1]: np.load(p) for p in glob(os.path.join(data_path, "*npy")) if "noises" in p}
    bgs = {p.split("/")[-1].split(".")[0].split("_")[-1]: np.load(p) for p in glob(os.path.join(data_path, "*npy")) if "background" in p}
    freqs = np.load(os.path.join(data_path, "freqs.npy"))


    diffs = {}
    for k, noise in noises.items():
        bg = bgs[k][None,...]
        diffs_it = []
        for noise_it in noise:
            diff = calculate_diffs(noise_it, bg)
            # print(noise_it.shape, bg.shape, diff.shape)
            diffs_it.append(diff)
            # break
        diffs_it = np.array(diffs_it)
        diffs[k] = diffs_it.mean(axis=0)
        # break


    diffs_a = diffs.pop("a")
    diffs_inv = {k:v/diffs_a for k, v in diffs.items()}

    k = list(diffs_inv.keys())[0]
    a, b, c, d1 = prepare_data(freqs[1:], diffs_inv[k][1:], x_gt, y1_gt)
    a, b, c, d2 = prepare_data(freqs[1:], diffs_inv[k][1:], x_gt, y2_gt)
    a, b, c, d3 = prepare_data(freqs[1:], diffs_inv[k][1:], x_gt, y3_gt)


    diffs_stack = np.stack([diffs_inv["3"][1:],
                            diffs_inv["6"][1:],
                            diffs_inv["12"][1:]])
    ds = np.stack([d1, d2, d3])

    order_corr = calculate_correlations_with_ground_truth(diffs_stack, ds)
    pearson_corr = pearson_correlation(diffs_stack.ravel(), ds.ravel())

    return {"ds": ds, "pearson_corr": pearson_corr, "kendall_corr": order_corr}

def download_data(data_path, # Path to download the data
                  ):
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    data_url = "https://zenodo.org/records/17700252/files/Experiment_5.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(data_path, "Experiment_5")
