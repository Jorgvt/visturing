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
from scipy.stats import pearsonr

from visturing.ranking import prepare_data, calculate_spearman
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


    c_mask = ['No mask', 'C_mask = 0.075', 'C_mask = 0.150', 'C_mask = 0.225', 'C_mask = 0.300']

    diffs_high = defaultdict(dict)
    for name, chroma in data_high.items():
        for f, dat in zip(c_mask, chroma):
            diffs_ = calculate_diffs(dat, dat[0:1])
            diffs_high[name][f] = diffs_

    diffs_low = defaultdict(dict)
    for name, chroma in data_low.items():
        for f, dat in zip(c_mask, chroma):
            diffs_ = calculate_diffs(dat, dat[0:1])
            diffs_low[name][f] = diffs_


    x_gt, y_low_gt, y_high_gt  = load_ground_truth(gt_path)


    diffs_low_a_s = np.array([a for a in diffs_low["achrom"].values()])
    diffs_low_rg_s = np.array([a for a in diffs_low["red_green"].values()])
    diffs_low_yb_s = np.array([a for a in diffs_low["yellow_blue"].values()])


    bs = []
    for b in diffs_low_a_s:
        a, b, c, d = prepare_data(xs, b, x_gt, y_low_gt)
        bs.append(b)
    b_low = np.array(bs)

    order_corr = {}
    order_corr["low"] = calculate_spearman(b_low, ideal_ordering=[0,1,2,3,4])


    diffs_high_a_s = np.array([a for a in diffs_high["achrom"].values()])

    bs = []
    for b in diffs_high_a_s:
        a, b, c, d = prepare_data(xs, b, x_gt, y_high_gt)
        bs.append(b)
    b_high = np.array(bs)

    order_corr["high"] = calculate_spearman(b_high, ideal_ordering=[0,1,2,3,4])

    pearson, p_value_pearson = pearsonr(
        np.concatenate([
            b_low[0], b_high[0]
        ]),
        np.concatenate([
            y_low_gt, y_high_gt
        ])
    )

    return {"diffs":
                {"low": diffs_low, "high": diffs_high},
            "correlations":
                {"kendall": order_corr, "pearson": pearson},
            "p_values":
                {"pearson": p_value_pearson}
            }

def download_data(data_path, # Path to download the data
                  ):
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    data_url = "https://zenodo.org/records/17700252/files/Experiment_8.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(data_path, "Experiment_8")

def generate_data(img_size: Sequence[int],
                  freq: Sequence[float],
                  freq_mask: Sequence[float],
                  L: float,
                  Cs: Sequence[float],
                  Cs_mask: float,
                  c: int, # 1 achrom 2 red-green 3 yellow-blue
                  fs: int,
                  theta: float,
                  theta_mask: float,
                  delta_theta: float = 0,
                  sigma_mask: float | None = None,
                  R0: float = 0,
                  n_iters: int = 1,
                  ):

    ## Generate the test
    stimuli = np.empty(shape=(n_iters, len(Cs), *img_size, 3))
    for i, C in enumerate(Cs):
        stimuli_, thetas = generate_noise_iters(img_size, freqs=freq, L=L, C=C, c=c, fs=fs, n_iters=n_iters, sigma_mask=sigma_mask, R0=R0, theta=theta, delta_theta=delta_theta)
        stimuli[:,i] = stimuli_[:,0]
    # stimuli = np.transpose(stimuli, axes=(1,0,2,3,4))

    ## Generate a masking background
    bgs = np.empty(shape=(len(Cs_mask), *img_size, 3))
    for i, C_mask in enumerate(Cs_mask):
        bg, theta_bg = generate_noise(img_size, fs=fs, freqs=freq_mask, L=L, C=C_mask, c=c, R0=R0, delta_theta=delta_theta, theta=theta_mask)
        bgs[i] = bg

    stimuli_bg = np.empty(shape=(n_iters, len(Cs_mask), len(Cs), *img_size, 3))
    for i, bg in enumerate(bgs):
        ## Add the mask to the test
        stimuli_bg[:,i] = stimuli + bg - bg.mean()

    ## Generate the plain image
    plain = generate_plain(img_size, L=L)

    ## Add the mask to the plain image
    plain = plain + bgs - bgs.mean()


    return stimuli_bg, plain, Cs_mask

def evaluate_gen(calculate_diffs,
                 img_size: Sequence[int],
                 freq: Sequence[float],
                 freq_mask: Sequence[float],
                 L: float,
                 Cs: Sequence[float],
                 Cs_mask: float,
                 fs: int,
                 sigma_mask: float | None = None,
                 theta: float = 0,
                 theta_mask: float = 0,
                 delta_theta: float = 0,
                 n_iters: int = 1,
                 return_stimuli: bool = False,
                 ):

    results = {}
    if return_stimuli:
        stimuli = {}
    for name, c in zip(["achrom", "red-green", "yellow-blue"], [1, 2, 3]):
        ## Generate ground truth
        stimuli_, plain_, freqs = generate_data(
                        img_size=img_size,
                        freq=freq,
                        freq_mask=freq_mask,
                        L=L,
                        Cs=Cs,
                        Cs_mask=Cs_mask,
                        c=c,
                        fs=fs,
                        sigma_mask=sigma_mask,
                        n_iters=n_iters,
                        theta=theta,
                        theta_mask=theta_mask,
                        delta_theta=delta_theta,
                        )

        if return_stimuli:
            stimuli[name] = stimuli_

        diffs = np.empty(shape=stimuli_.shape[:3])
        for i, stims in enumerate(stimuli_):
            for j, (s, plain) in enumerate(zip(stims, plain_)):
                diff = calculate_diffs(s, plain)
                diffs[i,j] = diff
                # fig, axes  = plt.subplots(2,10)
                # for k, ax in enumerate(axes[0].ravel()):
                #     ax.imshow(s[k])
                #     ax.set_title(f"{diff[k]:.3f}")
                # axes[1,5].imshow(plain)
                # for ax in axes.ravel(): ax.axis("off")
                # plt.show()

        diffs = diffs.mean(axis=0)
        results[name] = diffs

    if return_stimuli:
        return results, freqs, stimuli

    return results, freqs
