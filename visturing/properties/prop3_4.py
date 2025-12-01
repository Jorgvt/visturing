import os
from glob import glob
import wget
from zipfile import ZipFile

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as stats

from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth, calculate_correlations, prepare_and_correlate, prepare_and_correlate_order, calculate_spearman

def load_data(root_path: str):
    freqs = np.load(os.path.join(root_path, "freq.npy"))
    return freqs

def load_ground_truth(root_path: str = "../../ground_truth_decalogo", # Path to the root containing all the ground truth files
                      ): # Tuple (x, y, red-green, yellow-blue)
    data = sio.loadmat(os.path.join(root_path, "responses_CSF_achrom.mat"))
    data = data["CSF_achrom"]
    x, y = data
    data = sio.loadmat(os.path.join(root_path, "responses_CSF_RG.mat"))
    data = data["CSF_RG"]
    _, rg = data
    data = sio.loadmat(os.path.join(root_path, "responses_CSF_YB.mat"))
    data = data["CSF_YB"]
    _, yb = data
    return x, y, rg, yb

def plot_ground_truth(x,
                      y,
                      rg,
                      yb,
                      ): # Returns both the fig and axes objects
    fig, axes = plt.subplots()
    axes.plot(x, y, "k", label="Achromatic")
    axes.plot(x, rg, "r", label="Red-Green")
    axes.plot(x, yb, "b", label="Yellow-Blue")
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlim([1,32])
    axes.set_ylim(bottom=0.3)
    axes.legend()
    axes.set_xlabel("Frequency (cpd)")
    axes.set_ylabel("Sensitivity")
    return fig, axes


def evaluate(calculate_diffs,
             data_path,
             gt_path,
             ):

    ## Load ground truth
    x_gt, y_gt, rg_gt, yb_gt = load_ground_truth(gt_path)

    ## Load data
    noises = {p.split("/")[-1].split(".")[0].split("_")[-1]:np.load(p) for p in glob(os.path.join(data_path, "*")) if "noises" in p}
    bg = np.load(os.path.join(data_path, "background.npy"))

    ## Calculate the differences
    diffs = {}
    for k, noises_ in noises.items():
        diffs_it = []
        for noise_it in noises_:
            diff = calculate_diffs(noise_it, bg[None,...])
            # print(noise_it.shape, bg.shape, diff.shape)
            diffs_it.append(diff)
            # break
        diffs_it = np.array(diffs_it)
        diffs[k] = diffs_it.mean(axis=0)

    gt_s = np.stack([y_gt,
                    rg_gt,
                    yb_gt])


    diffs_s = np.stack([diffs["a"],
                        diffs["rg"],
                        diffs["yb"]])

    freqs = load_data(data_path)

    bs, ds = [], []
    for d, gt in zip(diffs_s, gt_s):
        a, b, c, d = prepare_data(freqs, d, x_gt, gt)
        bs.append(b)
        ds.append(d)
    b = np.array(bs)
    d = np.array(ds)

    order_corr = calculate_correlations_with_ground_truth(b, d)
    pearson_corr = stats.pearsonr(b.ravel(), d.ravel())

    return {"diffs_s": diffs_s, "pearson_corr": pearson_corr, "order_corr": order_corr}

def download_data(data_path, # Path to download the data
                  ):
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    data_url = "https://zenodo.org/records/17700252/files/Experiment_4_5.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(data_path, "Experiment_3_4")