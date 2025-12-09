import os
import re
from glob import glob
from collections import defaultdict
import wget
from zipfile import ZipFile


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as stats

from visturing.ranking import calculate_spearman

def load_ground_truth(root_path: str = "../../ground_truth_decalogo", # Path to the root containing all the ground truth files
                      return_freqs: bool = False, # Return the frequencies corresponding to each response
                      ): # Tuple (x, y1, y2, y3)
    data = sio.loadmat(os.path.join(root_path, "responses_no_mask_achrom_1p5_3_6_12_24.mat"))
    data = data["resp_no_mask_achrom"]
    x, y = data[0], data[1:]
    y_low, y_high = y[1], y[-2]
    freqs = [3, 12]
    
    if return_freqs:
        return x, y_low, y_high, freqs
    else:
        return x, y_low, y_high

def plot_ground_truth(x,
                      y_low,
                      y_high,
                      freqs=(3, 12),
                      figsize=(14,4),
                      ): # Returns both the fig and axes objects

    fig, axes = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,4))
    axes[0].plot(x, y_low, color="b", label=f"{freqs[0]} cpd")
    axes[1].plot(x, y_high, color="b", label=f"{freqs[1]} cpd")
    for ax in axes.ravel():
        ax.legend()
        ax.set_xlabel("Contrast")
    axes[0].set_ylabel("Visibility")
    return fig, axes

def evaluate(calculate_diffs,
             data_path,
             gt_path,
             ):
    
    if not os.path.exists(data_path):
        data_path = download_data("/".join(data_path.split("/")[:-1]))

    xs = np.load(os.path.join(data_path, "contrasts.npy"))
    data_high = {re.findall("high_(\w+)\.", p)[0]: np.load(p) for p in glob(os.path.join(data_path, "*")) if "high" in p}
    data_low = {re.findall("low_(\w+)\.", p)[0]: np.load(p) for p in glob(os.path.join(data_path, "*")) if "_low_" in p}


    f_mask = ['No mask', 'Theta_mask = 0', 'Theta_mask = 22.5', 'Theta_mask = 45', 'Theta_mask = 67.5', 'Theta_mask = 90', 'Theta_mask = 112.5', 'Theta_mask = 135']

    diffs_high = defaultdict(dict)
    for name, chroma in data_high.items():
        for f, dat in zip(f_mask, chroma):
            diffs_ = calculate_diffs(dat, dat[0:1])
            diffs_high[name][f] = diffs_

    diffs_low = defaultdict(dict)
    for name, chroma in data_low.items():
        for f, dat in zip(f_mask, chroma):
            diffs_ = calculate_diffs(dat, dat[0:1])
            diffs_low[name][f] = diffs_


    diffs_low_s = np.array([a for a in diffs_low["achrom"].values()])

    order_corr = {}
    order_corr["low"] = calculate_spearman(diffs_low_s, ideal_ordering=[0,7,6,5,3,1,2,4])

    diffs_high_s = np.array([a for a in diffs_high["achrom"].values()])
    order_corr["high"] = calculate_spearman(diffs_high_s, ideal_ordering=[0,7,6,5,3,1,2,4])

    return {"correlations":
                {"kendall": order_corr},
            "p_values":
                {}
            }

def download_data(data_path, # Path to download the data
                  ):
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    data_url = "https://zenodo.org/records/17700252/files/Experiment_10.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(data_path, "Experiment_10")
