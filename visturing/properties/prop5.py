from typing import Sequence
import os
from glob import glob
import wget
from zipfile import ZipFile

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth
from visturing.properties.noise import generate_noise_iters, generate_plain, generate_noise
from visturing.properties.formula import incremental_threshold_spatio_temp
from visturing.properties import prop3_4

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
    diffs_inv = {k:(diffs_a+1e-6)/v for k, v in diffs.items()}
    diffs_inv = {k:(v-1) for k, v in diffs_inv.items()}
    diffs_inv = {k:np.clip(v, a_min=1e-6, a_max=np.inf) for k, v in diffs_inv.items()}
    diffs_inv = {k:v/v.max() for k, v in diffs_inv.items()}

    k = list(diffs_inv.keys())[0]
    a, b, c, d1 = prepare_data(freqs[1:], diffs_inv[k][1:], x_gt, y1_gt)
    a, b, c, d2 = prepare_data(freqs[1:], diffs_inv[k][1:], x_gt, y2_gt)
    a, b, c, d3 = prepare_data(freqs[1:], diffs_inv[k][1:], x_gt, y3_gt)


    diffs_stack = np.stack([diffs_inv["3"][1:],
                            diffs_inv["6"][1:],
                            diffs_inv["12"][1:]])
    ds = np.stack([d1, d2, d3])

    order_corr = calculate_correlations_with_ground_truth(diffs_stack, ds)
    pearson_corr, p_value_pearson = pearsonr(diffs_stack.ravel(), ds.ravel())

    return {"ds": ds,
            "diffs": diffs_stack,
            "correlations":
                {"pearson": pearson_corr, "kendall": order_corr},
            "p_values":
                {"pearson": p_value_pearson},
        }

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

def generate_data(img_size: Sequence[int],
                  freqs: Sequence[int],
                  freqs_mask: Sequence[float],
                  L: float,
                  C: float,
                  C_mask: float,
                  c: int, # 1 achrom 2 red-green 3 yellow-blue
                  fs: int,
                  theta: float = 0,
                  theta_mask: float = 0,
                  delta_theta: float = 0,
                  sigma_mask: float | None = None,
                  R0: float = 0,
                  n_iters: int = 1,
                  ):

    stimuli, freqs = generate_noise_iters(img_size, freqs=freqs, L=L, C=C, c=c, fs=fs, n_iters=n_iters, sigma_mask=sigma_mask, R0=R0, theta=theta, delta_theta=delta_theta)

    ## Generate the mask
    bgs, freqs_bg = generate_noise(img_size, fs=fs, freqs=freqs_mask, L=L, C=C_mask, c=c, R0=R0, delta_theta=delta_theta, theta=theta_mask)

    stimuli_bg = np.empty(shape=(n_iters, len(freqs_mask), len(freqs), *img_size, 3))
    for i, bg in enumerate(bgs):
        ## Add the mask to the test
        stimuli_bg[:,i] = stimuli + bg[None,None,:] - bg.mean()

    plains = np.empty(shape=(len(freqs_mask), *img_size, 3))
    for i, bg in enumerate(bgs):
        ## Generate the plain image
        plain = generate_plain(img_size, L=L)

        ## Add the mask to the plain image
        plains[i] = plain + bg - bg.mean()

    return stimuli_bg, plains, freqs

def evaluate_gen(calculate_diffs,
                 img_size: Sequence[int],
                 freqs: Sequence[float],
                 freqs_mask: Sequence[float],
                 L: float,
                 C: float,
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

    ## Obtain the values for the non-masked CSF
    diffs_csf, _, _  = prop3_4.evaluate_gen(calculate_diffs,
                    img_size=img_size,
                    freqs=freqs,
                    L=L,
                    C=C,
                    fs=fs,
                    sigma_mask=sigma_mask,
                    n_iters=n_iters,
                    theta=theta,
                    delta_theta=delta_theta,
                    return_stimuli=False,
                    return_gt=False)
    print(diffs_csf)

    for name, c in zip(["achrom", "red-green", "yellow-blue"], [1, 2, 3]):
        ## Generate ground truth
        stimuli_, plain_, freqs = generate_data(img_size=img_size, freqs=freqs, freqs_mask=freqs_mask, L=L, C=C, C_mask=C_mask, c=c, fs=fs, sigma_mask=sigma_mask, n_iters=n_iters, theta=theta, delta_theta=delta_theta)
        if return_stimuli:
            stimuli[name] = stimuli_

        diffs = np.empty(shape=stimuli_.shape[:3])
        for i, stims in enumerate(stimuli_):
            for j, (s, plain) in enumerate(zip(stims, plain_)):
                # print(f"Stims: {s.shape}")
                # print(f"Plain: {plain.shape}")
                diff = calculate_diffs(s, plain)
                diffs[i,j] = diff

        diffs = diffs.mean(axis=0)
        results[name] = diffs

    ## Calculate the masked CSF
    csfs = {}
    for (name, csf), (name2, res)  in zip(diffs_csf.items(), results.items()):
        masked = np.empty_like(res)
        for i, csf_masked in enumerate(res):
            masked[i] = csf/csf_masked
        csfs[name] = masked
    results = csfs

    ## Get ground truth
    gts = {}
    for name, c in zip(["achrom", "red-green", "yellow-blue"], [1, 2, 3]):
        gt = get_ground_truth(
            freqs=freqs,
            freqs_mask=freqs_mask,
            C=[C],
            C_mask=[C_mask],
            c=3
        )
        gts[name] = gt

    res_flat = np.array([a.ravel() for a in csfs.values()]).ravel()
    gts_flat = np.array([a.ravel() for a in gts.values()]).ravel()

    correlation = pearsonr(res_flat, gts_flat)

    if return_stimuli:
        return results, freqs, stimuli, correlation

    return results, freqs, correlation

def get_ground_truth(
                    freqs: Sequence[float],
                    freqs_mask: Sequence[float],
                    C: Sequence[float],
                    C_mask: Sequence[float],
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
    cs_mask = np.array([C_mask])

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

    gt_csf = prop3_4.get_ground_truth(
        freqs=freqs,
        C=C,
        c=c
    )
    gt_csf = np.array(gt_csf)
    gt = np.empty(shape=(len(freqs_mask), len(gt_csf)))
    gt_sd = np.empty(shape=(len(freqs_mask), len(gt_csf)))
    for i, sup in enumerate(sups):
        gt[i] = gt_csf/sup[:,0]
        gt_sd[i] = sup[:,0]

    return gt
    # return sups[0][:,0]
