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
from visturing.properties.formula import incremental_threshold_spatio_temp
from .utils import EvaluationResult, run_batched
from .config import default_prop8_config as default_config


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
                  freqs: Sequence[float],
                  freqs_mask: Sequence[float],
                  L: float,
                  Cs: Sequence[float],
                  Cs_mask: Sequence[float],
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
    stimuli = np.empty(shape=(n_iters, len(freqs_mask), len(Cs_mask), len(freqs), len(Cs), *img_size, 3))
    for i, f_mask in enumerate(freqs_mask):
        for j, c_mask in enumerate(Cs_mask):
            for k, C in enumerate(Cs):
                stimuli_, thetas = generate_noise_iters(img_size, freqs=freqs, L=L, C=C, c=c, fs=fs, n_iters=n_iters, sigma_mask=sigma_mask, R0=R0, theta=theta, delta_theta=delta_theta)
                stimuli[:,i,j,:,k] = stimuli_

    ## Generate a masking background
    bgs = np.empty(shape=(len(freqs_mask), len(Cs_mask), *img_size, 3))
    for i, C_mask in enumerate(Cs_mask):
        bg, theta_bg = generate_noise(img_size, fs=fs, freqs=freqs_mask, L=L, C=C_mask, c=c, R0=R0, delta_theta=delta_theta, theta=theta_mask)
        bgs[:,i] = bg

    stimuli_bg = np.empty_like(stimuli)
    for i, bg_ in enumerate(bgs):
        for j, bg in enumerate(bg_):
            ## Add the mask to the test
            stimuli_bg[:,i,j] = stimuli[:,i,j] + bg - bg.mean()

    ## Generate the plain image
    plain = generate_plain(img_size, L=L)

    ## Add the mask to the plain image
    plain = plain + bgs - bgs.mean()


    return stimuli_bg, plain, Cs_mask

def evaluate_gen(calculate_diffs,
                 img_size: Sequence[int],
                 freqs: Sequence[float],
                 freqs_mask: Sequence[float],
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
                 return_gt: bool = False,
                 batch_size: int | None = None,
                 verbose: bool = False,
                 ):

    results = {}
    if return_stimuli:
        stimuli = {}
    for name, c in zip(["achrom", "red-green", "yellow-blue"], [1, 2, 3]):
        ## Generate ground truth
        stimuli_, plain_, _ = generate_data(
                        img_size=img_size,
                        freqs=freqs,
                        freqs_mask=freqs_mask,
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

        n_iters_val, num_f_mask, num_c_mask, num_freqs, num_cs, h, w, c_dim = stimuli_.shape
        stimuli_flat = stimuli_.reshape(-1, h, w, c_dim)

        plain_expanded = plain_[None, :, :, None, None, :, :, :]
        plain_expanded = np.repeat(plain_expanded, n_iters_val, axis=0)
        plain_expanded = np.repeat(plain_expanded, num_freqs, axis=3)
        plain_expanded = np.repeat(plain_expanded, num_cs, axis=4)
        plain_flat = plain_expanded.reshape(-1, h, w, c_dim)

        diffs_flat = run_batched(
            calculate_diffs, 
            stimuli_flat, 
            plain_flat, 
            batch_size=batch_size,
            show_progress=verbose,
            desc=f"prop8 ({name})"
        )
        diff = diffs_flat.reshape(n_iters_val, num_f_mask, num_c_mask, num_freqs, num_cs)

        diffs = diff.mean(axis=0)
        results[name] = diffs

    for name, r in results.items():
        print(f"{name}: {r.shape}")

    ## Get ground truth to calculate correlation
    gts = {}
    for name, c in zip(["achrom", "red-green", "yellow-blue"], [1, 2, 3]):
        gt, Z = get_ground_truth(
            freqs=freqs,
            freqs_mask=freqs_mask,
            C=Cs,
            Cs_mask=Cs_mask,
            c=c
        )
        ## Skip 0s as of now
        gts[name] = Z[...,1:]
 
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
                    freqs_mask: Sequence[float],
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

    fs_mask = freqs_mask
    cs_mask = Cs_mask

    sups = np.empty(shape=(len(fs_mask), len(cs_mask), len(fs_test), len(cs_test)))
    for ii, fm in enumerate(fs_mask):
        for jj, Cm in enumerate(cs_mask):
            S_malo = np.zeros((len(fs_test), len(cs_test)))
            # --- Calcular sensibilidad con el masker ---
            for i, f_val in enumerate(fs_test):
                for j, C_val in enumerate(cs_test):
                    Delta_C, _, _, _, _ = incremental_threshold_spatio_temp(
                        f_val, 0, 0, C_val, fm, 0, 0, Cm, c, kind 
                    )
                    S_malo[i,j] = 1 / Delta_C
            # sups.append(S_malo)
            sups[ii,jj] = S_malo

    print(f"Sups: {sups.shape}")
    ## Include 0s and cumsum to obtain response curves
    Zs = np.empty(shape=(len(fs_mask), len(cs_mask), len(fs_test), len(cs_test)+1))
    for i, f_mask in enumerate(freqs_mask):
        for j, c_mask in enumerate(Cs_mask):
            # Generate data
            Cs_ = np.insert(C, 0, 0)
            A, B = np.meshgrid(freqs, Cs_)
            Z_ = sups[i,j].T
            Z = np.zeros(shape=(Z_.shape[0]+1, Z_.shape[1]))
            Z[1:] = Z_

            Z = Z.cumsum(axis=0)
            Zs[i,j] = Z.T # Sus

    return sups, Zs
