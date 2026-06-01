import os
from typing import Sequence
from glob import glob
import wget
from zipfile import ZipFile

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth
from visturing.properties.noise import generate_noise_iters, generate_plain
from visturing.properties.formula import incremental_threshold_spatio_temp

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

    if not os.path.exists(data_path):
        data_path = download_data("/".join(data_path.split("/")[:-1]))

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
    pearson_corr, p_value_pearson = pearsonr(b.ravel(), d.ravel())

    return {"diffs_s": diffs_s,
            "correlations":
                {"pearson": pearson_corr, "kendall": order_corr},
            "p_values": 
                {"pearson": p_value_pearson},
        }

def download_data(data_path, # Path to download the data
                  ):
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    data_url = "https://zenodo.org/records/17700252/files/Experiment_3_4.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(data_path, "Experiment_3_4")

def generate_data(img_size: Sequence[int],
                  freqs: Sequence[int],
                  L: float,
                  C: float,
                  c: int, # 1 achrom 2 red-green 3 yellow-blue
                  fs: int,
                  theta: float = 0,
                  delta_theta: float = 0,
                  sigma_mask: float | None = None,
                  R0: float = 0,
                  n_iters: int = 1,
                  ):

    stimuli, freqs = generate_noise_iters(img_size, freqs=freqs, L=L, C=C, c=c, fs=fs, n_iters=n_iters, sigma_mask=sigma_mask, R0=R0, theta=theta, delta_theta=delta_theta)

    ## Generate the plain image
    plain = generate_plain(img_size, L=L)

    return stimuli, plain, freqs

def evaluate_gen(calculate_diffs,
                 img_size: Sequence[int],
                 freqs: Sequence[float],
                 L: float,
                 C: float,
                 fs: int,
                 sigma_mask: float | None = None,
                 theta: float = 0,
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
        stimuli_, plain, freqs = generate_data(img_size=img_size, freqs=freqs, L=L, C=C, c=c, fs=fs, sigma_mask=sigma_mask, n_iters=n_iters, theta=theta, delta_theta=delta_theta)
        if return_stimuli:
            stimuli[name] = stimuli_

        diffs = np.empty(shape=stimuli_.shape[:2])
        for i, stims in enumerate(stimuli_):
            diff = calculate_diffs(stims, plain)
            diffs[i] = diff

        diffs = diffs.mean(axis=0)
        results[name] = diffs


    gt = {}
    for name, res in results.items():
        if name == "achrom": c = 1
        elif name == "red-green": c = 2
        elif name == "yellow-blue": c = 3

        gt_ = get_ground_truth(freqs=freqs, C=[C], c=c)
        gt[name] = gt_

    ## Correlations have to be calculated all together
    correlations = {}
    preds = np.stack([a for a in results.values()]).ravel()
    gts = np.stack([a for a in gt.values()]).ravel()
    correlations["pearson"] = pearsonr(preds, gts)[0]


    if return_stimuli and return_gt:
        return results, freqs, stimuli, correlations, gt
    elif return_stimuli and not return_gt:
        return results, freqs, stimuli, correlations
    if not return_stimuli and return_gt:
        return results, freqs, correlations, gt

    return results, freqs, correlations

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

    fs_mask = np.array([0])
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

    return sups[0][:,0]
