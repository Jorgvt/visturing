import enum
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

from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth
from visturing.properties.noise import generate_noise_iters, generate_plain, generate_noise
from visturing.properties.formula import incremental_threshold_spatio_temp
from visturing.properties.prob_weight import get_weights
from .utils import EvaluationResult, run_batched, weighted_pearson_correlation, ArrayApi
from .config import default_prop6_7_config as default_config


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

    if not os.path.exists(data_path):
        data_path = download_data("/".join(data_path.split("/")[:-1]))

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
    pearson, p_value_pearson = pearsonr(
        b_cat[~nan_mask], d_cat[~nan_mask]
    )

    return {"diffs": diffs,
            "correlations":
                {"kendall": order_corr, "pearson": pearson},
            "p_values":
                {"pearson": p_value_pearson}
            }

def download_data(data_path, # Path to download the data
                  ):
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    data_url = "https://zenodo.org/records/17700252/files/Experiment_6_7.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(data_path, "Experiment_6_7")

def generate_data(img_size: Sequence[int],
                  freqs: Sequence[float],
                  L: float,
                  Cs: Sequence[float],
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

    ## Generate the plain image
    plain = generate_plain(img_size, L=L)

    return stimuli, plain, freqs

def evaluate_gen(calculate_diffs,
                 img_size: Sequence[int],
                 freqs: Sequence[float],
                 L: float,
                 Cs: Sequence[float],
                 fs: int,
                 sigma_mask: float | None = None,
                 theta: float = 0,
                 delta_theta: float = 0,
                 n_iters: int = 1,
                 return_stimuli: bool = False,
                 return_gt: bool = False,
                 batch_size: int | None = None,
                 verbose: bool = False,
                 xp=np,
                 ):

    xp_api = ArrayApi(xp)

    results = {}
    if return_stimuli:
        stimuli = {}
    for name, c in zip(["achrom", "red-green", "yellow-blue"], [1, 2, 3]):
        ## Generate ground truth
        stimuli_, plain, freqs = generate_data(
                        img_size=img_size,
                        freqs=freqs,
                        L=L,
                        Cs=Cs,
                        c=c,
                        fs=fs,
                        sigma_mask=sigma_mask,
                        n_iters=n_iters,
                        theta=theta,
                        delta_theta=delta_theta,
                        )

        if return_stimuli:
            stimuli[name] = stimuli_

        n_iters_val, num_cs, num_freqs, h, w, c_dim = stimuli_.shape
        stimuli_flat = stimuli_.reshape(-1, h, w, c_dim)
        plain_flat = np.repeat(plain, n_iters_val * num_cs * num_freqs, axis=0)

        diffs_flat = run_batched(
            calculate_diffs, 
            stimuli_flat, 
            plain_flat, 
            batch_size=batch_size,
            show_progress=verbose,
            desc=f"prop6_7 ({name})"
        )
        diff = diffs_flat.reshape(n_iters_val, num_cs, num_freqs)

        diffs = xp_api.mean(diff, axis=0)
        results[name] = diffs

    for name, r in results.items():
        print(f"{name}: {r.shape}")

    ## Get ground truth to calculate correlation
    gts = {}
    for name, c in zip(["achrom", "red-green", "yellow-blue"], [1, 2, 3]):
        gt, Z = get_ground_truth(
            freqs=freqs,
            C=Cs,
            c=c
        )
        ## Skip 0s as of now
        gts[name] = Z[1:]
 
    ## Obtain the weight following the probability of the images
    weights = get_weights(freqs=freqs, Cs=Cs, Bpp=3)

    ## Correlations have to be calculated all together
    correlations = {"non-weighted": {}, "weighted": {}}
    preds = xp_api.ravel(xp_api.stack([results[k] for k in results.keys()]))
    gts_flat = xp_api.asarray(np.stack([gts[k] for k in results.keys()]).ravel())
    weights_global = xp_api.asarray(np.stack([weights[k] for k in results.keys()]).ravel())

    correlations["non-weighted"]["global"] = weighted_pearson_correlation(preds, gts_flat, xp_api.ones_like(weights_global), xp=xp_api)
    correlations["weighted"]["global"] = weighted_pearson_correlation(preds, gts_flat, weights_global, xp=xp_api)

    for name in results.keys():
        r_name = results[name]
        gt_name = xp_api.asarray(gts[name])
        w_name = xp_api.asarray(weights[name])
        correlations["non-weighted"][name] = weighted_pearson_correlation(xp_api.ravel(r_name), xp_api.ravel(gt_name), xp_api.ones_like(xp_api.ravel(w_name)), xp=xp_api)
        correlations["weighted"][name] = weighted_pearson_correlation(xp_api.ravel(r_name), xp_api.ravel(gt_name), xp_api.ravel(w_name), xp=xp_api)

    return EvaluationResult(
        results=results,
        correlations=correlations,
        stimuli=stimuli if return_stimuli else None,
        gt=gts if return_gt else None,
        freqs=freqs,
    )


def get_ground_truth(
                    freqs: Sequence[float],
                    C: Sequence[float],
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

    fs_mask = np.array([0.])
    cs_mask = np.array([0.])

    sups = []
    for fm in fs_mask:
        Cm = cs_mask
        S_malo = np.zeros((len(fs_test), len(cs_test)))
        # --- Calcular sensibilidad con el masker ---
        for i, f_val in enumerate(fs_test):
            for j, C_val in enumerate(cs_test):
                Delta_C, _, _, _, _ = incremental_threshold_spatio_temp(
                    f_val, 0, 0, C_val, fm, 0, 0, Cm, c, kind 
                )
                S_malo[i,j] = 1 / Delta_C
        sups.append(S_malo)

    sups = np.array(sups)
    print(f"Sups: {sups.shape}")

    ## Include 0s and cumsum to obtain response curves
    Cs_ = np.insert(C, 0, 0)
    A, B = np.meshgrid(freqs, Cs_)
    Z_ = sups[0].T
    Z = np.zeros(shape=(Z_.shape[0]+1, Z_.shape[1]))
    Z[1:] = Z_
    Z = Z.cumsum(axis=0)


    return sups, Z
    # return sups[0][:,0]
