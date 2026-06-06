import enum
from  typing import Sequence
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

from visturing.ranking import calculate_spearman
from visturing.properties.noise import generate_noise, generate_noise_iters, generate_plain
from visturing.properties.formula import incremental_threshold_spatio_temp
from .utils import EvaluationResult


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
    order_low_1 = calculate_spearman(diffs_low_s[:5], ideal_ordering=[4,3,2,1,0])
    order_low_2 = calculate_spearman(diffs_low_s[4:], ideal_ordering=[0,1,2])
    order_corr["low"] = {k1:(v1*5+v2*3)/8 for (k1,v1), (k2,v2) in zip(order_low_1.items(), order_low_2.items())}

    diffs_high_s = np.array([a for a in diffs_high["achrom"].values()])
    order_high_1 = calculate_spearman(diffs_high_s[:5], ideal_ordering=[4,3,2,1,0])
    order_high_2 = calculate_spearman(diffs_high_s[4:], ideal_ordering=[0,1,2])
    order_corr["high"] = {k1:(v1*5+v2*3)/8 for (k1,v1), (k2,v2) in zip(order_high_1.items(), order_high_2.items())}

    return {"diffs":
                {"low": diffs_low_s,
                 "high": diffs_high_s},
            "correlations":
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

def generate_data(img_size: Sequence[int],
                  freqs: Sequence[float],
                  freq_mask: float,
                  L: float,
                  Cs: Sequence[float],
                  Cs_mask: Sequence[float],
                  c: int, # 1 achrom 2 red-green 3 yellow-blue
                  fs: int,
                  theta: float,
                  thetas_mask: Sequence[float],
                  delta_theta: float = 0,
                  sigma_mask: float | None = None,
                  R0: float = 0,
                  n_iters: int = 1,
                  ):

    ## Generate the test
    stimuli = np.empty(shape=(n_iters, len(thetas_mask), len(Cs_mask), len(freqs), len(Cs), *img_size, 3))
    for i, theta_mask in enumerate(thetas_mask):
        for j, c_mask in enumerate(Cs_mask):
            for k, C in enumerate(Cs):
                stimuli_, thetas = generate_noise_iters(img_size, freqs=freqs, L=L, C=C, c=c, fs=fs, n_iters=n_iters, sigma_mask=sigma_mask, R0=R0, theta=theta, delta_theta=delta_theta)
                stimuli[:,i,j,:,k] = stimuli_

    ## Generate a masking background
    bgs = np.empty(shape=(len(thetas_mask), len(Cs_mask), len(freqs), *img_size, 3))
    for i, theta_mask in enumerate(thetas_mask):
        for j, C_mask in enumerate(Cs_mask):
            bg, theta_bg = generate_noise(img_size, fs=fs, freqs=freqs, L=L, C=C_mask, c=c, R0=R0, delta_theta=delta_theta, theta=theta_mask)
            bgs[i,j] = bg

    stimuli_bg = np.empty_like(stimuli)
    for i, bg_ in enumerate(bgs):
        for j, bg in enumerate(bg_):
            ## Add the mask to the test
            stimuli_bg[:,i,j] = stimuli[:,i,j] + bg[None,:,None,...] - bg.mean()

    ## Generate the plain image
    plain = generate_plain(img_size, L=L)

    ## Add the mask to the plain image
    plain = plain + bgs - bgs.mean()


    return stimuli_bg, plain, thetas_mask

def evaluate_gen(calculate_diffs,
                 img_size: Sequence[int],
                 freqs: Sequence[float],
                 freq_mask: Sequence[float],
                 L: float,
                 Cs: Sequence[float],
                 Cs_mask: float,
                 fs: int,
                 sigma_mask: float | None = None,
                 theta: float = 0,
                 thetas_mask: Sequence[float] = [0],
                 delta_theta: float = 0,
                 n_iters: int = 1,
                 return_stimuli: bool = False,
                 return_gt: bool = False,
                 ):

    results = {}
    if return_stimuli:
        stimuli = {}
    for name, c in zip(["achrom", "red-green", "yellow-blue"], [1, 2, 3]):
        ## Generate ground truth
        stimuli_, plain_, _ = generate_data(
                        img_size=img_size,
                        freqs=freqs,
                        freq_mask=freq_mask,
                        L=L,
                        Cs=Cs,
                        Cs_mask=Cs_mask,
                        c=c,
                        fs=fs,
                        sigma_mask=sigma_mask,
                        n_iters=n_iters,
                        theta=theta,
                        thetas_mask=thetas_mask,
                        delta_theta=delta_theta,
                        )
        print(f"Stimuli_: {stimuli_.shape}")

        if return_stimuli:
            stimuli[name] = stimuli_

        # diffs = np.empty(shape=stimuli_.shape[:5])
        # for i, stims in enumerate(stimuli_):
        #     for j, (s, plain) in enumerate(zip(stims, plain_)):
        #         # print(f"Stims: {stims.shape}")
        #         # print(f"Plain: {plain.shape}")
        #         diff = calculate_diffs(s, plain[None,:,:,None])
        #         # print(f"Diff: {diff.shape}")
        #         diffs[i,j] = diff
        #         # fig, axes  = plt.subplots(2,10)
        #         # for k, ax in enumerate(axes[0].ravel()):
        #         #     ax.imshow(s[k])
        #         #     ax.set_title(f"{diff[k]:.3f}")
        #         # axes[1,5].imshow(plain)
        #         # for ax in axes.ravel(): ax.axis("off")
        #         # plt.show()

        diffs = np.empty(shape=stimuli_.shape[:5])
        for i, stims in enumerate(stimuli_):
            for j, (s, plain) in enumerate(zip(stims, plain_)):
                for k, (s_, plain__) in enumerate(zip(s, plain)):
                    diff = calculate_diffs(s_, plain__[:])
                    diffs[i,j,k] = diff
        diffs = diffs.mean(axis=0)
        results[name] = diffs

    ## Get ground truth to calculate correlation
    gts = {}
    for name, c in zip(["achrom", "red-green", "yellow-blue"], [1, 2, 3]):
        gt = get_ground_truth(
            freqs=freqs,
            thetas_mask=thetas_mask,
            C=Cs,
            Cs_mask=Cs_mask,
            c=c
        )
        ## Skip 0s as of now
        gts[name] = gt
 
    res_flat = np.array([a.ravel() for a in results.values()]).ravel()
    gts_flat = np.array([a.ravel() for a in gts.values()]).ravel()

    correlation = {}
    for (name, res), (name, gt_) in zip(results.items(), gts.items()):
        correlation[name] = pearsonr(res.ravel(), gt_.ravel())
    correlation["global"] = pearsonr(res_flat, gts_flat)

    return EvaluationResult(
        results=results,
        correlations=correlation,
        stimuli=stimuli if return_stimuli else None,
        gt=gts if return_gt else None,
        freqs=freqs,
    )

def get_ground_truth(
                    freqs: Sequence[float],
                    thetas_mask: Sequence[float],
                    C: Sequence[float],
                    Cs_mask: Sequence[float],
                    c: int,
                    ):

    fs_test = freqs
    cs_test = C
    if c == 1:
        kind = 1
    elif c == 2:
        kind = 4
    elif c == 3:
        kind = 4

    fs_mask = freqs
    cs_mask = Cs_mask

    sups = np.empty(shape=(len(freqs), len(cs_mask), len(thetas_mask), len(cs_test)))
    for ii, f_val in enumerate(freqs):
        for jj, Cm in enumerate(cs_mask):
            S_malo = np.zeros((len(thetas_mask), len(cs_test)))
            # --- Calcular sensibilidad con el masker ---
            for i, tm in enumerate(thetas_mask):
                for j, C_val in enumerate(cs_test):
                    fm = f_val
                    tm = tm*np.pi/180
                    fmx = np.cos(tm)*fm
                    fmy = np.sin(tm)*fm
                    Delta_C, _, _, _, _ = incremental_threshold_spatio_temp(
                        f_val, 0, 0, C_val, fmx, fmy, 0, Cm, c, kind 
                    )
                    S_malo[i,j] = 1 / Delta_C
            # sups.append(S_malo)
            sups[ii,jj] = S_malo

    return sups
