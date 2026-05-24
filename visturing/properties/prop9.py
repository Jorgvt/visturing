from typing import Sequence
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
from visturing.properties.noise import generate_noise, generate_noise_iters, generate_plain

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


    f_mask = ['No mask', 'F_mask = 1.5 cpd', 'F_mask = 3 cpd', 'F_mask = 6 cpd', 'F_mask = 12 cpd', 'F_mask = 24 cpd']

    diffs_high = defaultdict(dict)
    for name, chroma in data_high.items():
        for f, dat in zip(f_mask, chroma):
            if f == "No mask": continue
            diffs_ = calculate_diffs(dat, dat[0:1])
            diffs_high[name][f] = diffs_

    diffs_low = defaultdict(dict)
    for name, chroma in data_low.items():
        for f, dat in zip(f_mask, chroma):
            if f == "No mask": continue
            diffs_ = calculate_diffs(dat, dat[0:1])
            diffs_low[name][f] = diffs_

    order_corr = {}

    diffs_low_s = np.array([a for a in diffs_low["achrom"].values()])
    order_low_1 = calculate_spearman(diffs_low_s[:2], ideal_ordering=[0,1])
    order_low_2 = calculate_spearman(diffs_low_s[1:], ideal_ordering=[3,2,1,0])
    order_corr["low"] = {k1:(v1*2+v2*4)/6 for (k1,v1), (k2,v2) in zip(order_low_1.items(), order_low_2.items())}

    diffs_high_s = np.array([a for a in diffs_high["achrom"].values()])
    order_high_1 = calculate_spearman(diffs_high_s[:2], ideal_ordering=[0,1])
    order_high_2 = calculate_spearman(diffs_high_s[1:], ideal_ordering=[3,2,1,0])
    order_corr["high"] = {k1:(v1*2+v2*4)/6 for (k1,v1), (k2,v2) in zip(order_high_1.items(), order_high_2.items())}

    return {"diffs":
                {"low": diffs_low,
                 "high": diffs_high},
            "correlations":
                {"kendall": order_corr},
            "p_values":
                {}
            }

def download_data(data_path, # Path to download the data
                  ):
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    data_url = "https://zenodo.org/records/17700252/files/Experiment_9.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(data_path, "Experiment_9")


def generate_data(img_size: Sequence[int],
                  freqs: Sequence[float],
                  freqs_mask: Sequence[float],
                  L: float,
                  Cs: Sequence[float],
                  C_mask: float,
                  c: int, # 1 achrom 2 red-green 3 yellow-blue
                  fs: int,
                  theta: float = 0,
                  delta_theta: float = 0,
                  sigma_mask: float | None = None,
                  R0: float = 0,
                  n_iters: int = 1,
                  ):

    ## Generate the test
    stimuli = np.empty(shape=(len(Cs), n_iters, len(freqs), *img_size, 3))
    for i, C in enumerate(Cs):
        stimuli_, freqs = generate_noise_iters(img_size, freqs=freqs, L=L, C=C, c=c, fs=fs, n_iters=n_iters, sigma_mask=sigma_mask, R0=R0, theta=theta, delta_theta=delta_theta)
        stimuli[i] = stimuli_
    stimuli = np.transpose(stimuli, axes=(1,0,2,3,4,5))

    ## Generate a masking background
    bg, f_bg = generate_noise(img_size, fs=fs, freqs=freqs_mask, L=L, C=C_mask, c=c, R0=R0, delta_theta=delta_theta, theta=theta)

    ## Add the mask to the test
    stimuli = stimuli + bg[None,None,:] - bg.mean()

    ## Generate the plain image
    plain = generate_plain(img_size, L=L)

    ## Add the mask to the plain image
    plain = plain + bg - bg.mean()


    return stimuli, plain, freqs

def evaluate_gen(calculate_diffs,
                 img_size: Sequence[int],
                 freqs: Sequence[float],
                 freqs_mask: Sequence[float],
                 L: float,
                 Cs: Sequence[float],
                 C_mask: float,
                 fs: int,
                 sigma_mask: float | None = None,
                 theta: float = 0,
                 delta_theta: float = 0,
                 n_iters: int = 1,
                 return_stimuli: bool = False,
                 ):

    results = {}
    if return_stimuli:
        stimuli = {}
    for name, c in zip(["achrom", "red-green", "yellow-blue"], [1, 2, 3]):
        ## Generate ground truth
        stimuli_, plain, freqs = generate_data(
                        img_size=img_size,
                        freqs=freqs,
                        freqs_mask=freqs_mask,
                        L=L,
                        Cs=Cs,
                        C_mask=C_mask,
                        c=c,
                        fs=fs,
                        sigma_mask=sigma_mask,
                        n_iters=n_iters,
                        theta=theta,
                        delta_theta=delta_theta,
                        )

        if return_stimuli:
            stimuli[name] = stimuli_

        diffs = np.empty(shape=stimuli_.shape[:3])
        for i, stims in enumerate(stimuli_):
            for j, s in enumerate(stims):
                diff = calculate_diffs(s, plain)
                diffs[i,j] = diff

        diffs = diffs.mean(axis=0)
        results[name] = diffs

    if return_stimuli:
        return results, freqs, stimuli

    return results, freqs
