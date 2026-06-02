import os
from glob import glob
from typing import Sequence
import wget
from zipfile import ZipFile

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from visturing.ranking import prepare_data, calculate_spearman
from perceptualtests.color_matrices import Mxyz2ng, gamma

def load_ground_truth(data_path: str = "ground_truth_decalogo", # Path to the root containing all the ground truth files
                      ): # Tuple of tuples (x, y), (x_c, red-green), (x_c, yellow-blue)
    data = sio.loadmat(os.path.join(data_path, "weber.mat"))
    data = data["weber"]

    x = data[0]
    y = data[1]

    data = sio.loadmat(os.path.join(data_path, "resp_RG.mat"))
    data = data["resp_RG"]
    x_c, rg, _ = data

    data = sio.loadmat(os.path.join(data_path, "resp_YB.mat"))
    data = data["resp_YB"]
    _, yb = data
    
    return x, y, x_c, rg, x_c, yb

def load_data(data_path):
    x_a = np.load(os.path.join(data_path, "luminancias.npy"))
    x_rg = np.load(os.path.join(data_path, "x_rg.npy"))
    x_yb = np.load(os.path.join(data_path, "x_yb.npy"))
    return x_a, x_rg, x_yb

def load_images(data_path):
    data = {p.split("/")[-1].split(".")[0]: np.load(p, allow_pickle=True) for p in glob(os.path.join(data_path, "*npy")) if "bgs" not in p and "luminancias" not in p and "x_rg" not in p and "x_yb" not in p}
    bgs = {p.split("/")[-1].split(".")[0][4:]: np.load(p, allow_pickle=True) for p in glob(os.path.join(data_path, "*npy")) if "bgs" in p}
    return data, bgs


def plot_ground_truth(x,y,
                      x_c, rg,
                      x_cc, yb,
                      ): # Returns both the fig and axes objects
    fig, axes = plt.subplots(1,3, figsize=(18,5))
    axes[0].plot(x, y, "b")
    axes[1].plot(x_c, rg, "gray")
    axes[2].plot(x_c, yb, "gray")
    for ax, name in zip(axes, ["Achromatic", "Red-Green", "Yellow-Blue"]): ax.set_title(name)
    axes[0].set_xlabel(r"Luminance (cd/m$^2$)")
    axes[1].set_xlabel("Linear RG")
    axes[2].set_xlabel("Linear YB")
    for ax, name in zip(axes, ["Brightness", "Nonlinear RG", "Nonlinear YB"]): ax.set_ylabel(name)
    axes[1].set_xlim([-22,22])
    axes[1].set_ylim([-8,8])
    axes[2].set_xlim([-22,22])
    axes[2].set_ylim([-8,8])
    return fig, axes

def evaluate(calculate_diffs,
             data_path: str = "Data/Experiment_2",
             gt_path: str = "ground_truth_decalogo",
             ): # Tuple (responses, correlations)

    if not os.path.exists(data_path):
        data_path = download_data("/".join(data_path.split("/")[:-1]))

    ## Load ground truth
    x_a_gt, y_a_gt, x_rg_gt, y_rg_gt, x_yb_gt, y_yb_gt = load_ground_truth(gt_path)

    ## Load data
    x_a, x_rg, x_yb = load_data(data_path)

    data = {p.split("/")[-1].split(".")[0]: np.load(p) for p in glob(os.path.join(data_path, "*npy")) if "bgs" not in p}
    bgs = {p.split("/")[-1].split(".")[0][4:]: np.load(p) for p in glob(os.path.join(data_path, "*npy")) if "bgs" in p}

    diffs = {}
    for c in ["achrom"]:
        diffs[c] = []
        data_ = data[c]
        bgs_ = bgs[c]
        for cc, bg in zip(data_, bgs_):
            diff = calculate_diffs(cc, cc[0:1])
            diffs[c].append(diff)
    for c in ["red_green", "yellow_blue"]:
        diffs[c] = []
        data_ = data[c]
        bgs_ = bgs[c]
        for cc, bg in zip(data_, bgs_):
            bg_idx = np.argwhere(np.where(cc==bg, True, False).all(axis=(1,2,3))).squeeze()
            diff = calculate_diffs(cc, bg[None,...])
            bg_mask = -1*np.ones_like(diff)
            bg_mask[bg_idx:] = 1
            diff = bg_mask*diff
            diffs[c].append(diff)
    diffs = {k: np.array(v) for k, v in diffs.items()}

    ## Calculate Order Correlation
    spearman_correlations = {}

    a, b, c, d = prepare_data(x_a, diffs["achrom"], x_a_gt, y_a_gt)
    spearman_correlations["achrom"] = calculate_spearman(b, ideal_ordering=[0,1,2,3,4])

    a, b_rg, c, d_rg = prepare_data(x_rg, diffs["red_green"], x_rg_gt, y_rg_gt)
    spearman_correlations["red_green"] = calculate_spearman(b_rg, ideal_ordering=[0,1,2,3,4])

    a, b_yb, c, d_yb = prepare_data(x_yb, diffs["yellow_blue"], x_yb_gt, y_yb_gt)
    spearman_correlations["yellow_blue"] = calculate_spearman(b_yb, ideal_ordering=[0,1,2,3,4])

    # Calculate Pearson
    ## Achromatic

    corr_achrom, p_value_achrom = pearsonr(
        np.concatenate([
            b[0].ravel(),
        ]),
        np.concatenate([
            d.ravel(),
        ])
    )
    ## Both Chromatics Together
    corr_chroma, p_value_chroma = pearsonr(
        np.concatenate([
            b_rg[2].ravel(), b_yb[2].ravel(),
        ]),
        np.concatenate([
            d_rg.ravel(), d_yb.ravel(),
        ])
    )
    correlations = {"pearson_achrom": corr_achrom, "pearson_chrom": corr_chroma, "kendall": spearman_correlations}

    return {"diffs": diffs,
            "correlations": correlations,
            "p_values":
                {"achrom": p_value_achrom,
                 "red_green": p_value_chroma},
            }

def download_data(data_path, # Path to download the data
                  ):
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    data_url = "https://zenodo.org/records/17700252/files/Experiment_2.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(data_path, "Experiment_2")

def xyz2lab(C, W):
    """
    XYZ2LAB computes the lightness, L*, and the chromatic coordinates a* and b* in CIELAB
    of a set of colours defined by their tristimulus CIE-1931 values.

    C: Tristimulus values of the test stimuli (Nx3 or 1x3 matrix).
    W: Tristimulus values of the reference stimulus (1x3 or Nx3 matrix).
    """
    C = np.array(C)
    W = np.array(W)
    
    is_1d = C.ndim == 1
    if is_1d:
        C = C[np.newaxis, :]
    
    if W.ndim == 1:
        W = W[np.newaxis, :]

    if W.shape[0] != C.shape[0]:
        W = np.tile(W, (C.shape[0], 1))

    ratio = C / W
    con = ratio > 0.008856
    
    # F calculation
    F = np.zeros_like(ratio)
    F[con] = ratio[con]**(1/3)
    F[~con] = (903.3 * ratio[~con] + 16) / 116

    Lab = np.zeros_like(C)
    Lab[:, 0] = 116 * F[:, 1] - 16
    Lab[:, 1] = 500 * (F[:, 0] - F[:, 1])
    Lab[:, 2] = 200 * (F[:, 1] - F[:, 2])

    if is_1d:
        return Lab[0]
    return Lab

def deltaE2000(Labstd, Labsample, KLCH=None):
    """
    Compute the CIEDE2000 color-difference between the sample and a standard.
    Labstd and Labsample are K x 3 matrices or 1 x 3 vectors.
    KLCH is an optional 1x3 vector containing kL, kC, and kH (default [1, 1, 1]).
    Based on the article:
    "The CIEDE2000 Color-Difference Formula: Implementation Notes, 
    Supplementary Test Data, and Mathematical Observations,", G. Sharma, 
    W. Wu, E. N. Dalal, Color Research and Application, vol. 30. No. 1, pp.
    21-30, February 2005.
    available at http://www.ece.rochester.edu/~/gsharma/ciede2000/
    https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/
    """
    Labstd = np.array(Labstd)
    Labsample = np.array(Labsample)

    is_1d = Labstd.ndim == 1 and Labsample.ndim == 1
    
    if Labstd.ndim == 1:
        Labstd = Labstd[np.newaxis, :]
    if Labsample.ndim == 1:
        Labsample = Labsample[np.newaxis, :]

    # Handle broadcasting if one is single row and other is multiple rows
    if Labstd.shape[0] == 1 and Labsample.shape[0] > 1:
        Labstd = np.tile(Labstd, (Labsample.shape[0], 1))
    elif Labsample.shape[0] == 1 and Labstd.shape[0] > 1:
        Labsample = np.tile(Labsample, (Labstd.shape[0], 1))

    if Labstd.shape != Labsample.shape:
        raise ValueError("deltaE00: Standard and Sample sizes do not match")
    if Labstd.shape[1] != 3:
        raise ValueError("deltaE00: Standard and Sample Lab vectors should be Kx3 vectors")

    if KLCH is None:
        kl, kc, kh = 1.0, 1.0, 1.0
    else:
        kl, kc, kh = KLCH

    Lstd = Labstd[:, 0]
    astd = Labstd[:, 1]
    bstd = Labstd[:, 2]
    Cabstd = np.sqrt(astd**2 + bstd**2)

    Lsample = Labsample[:, 0]
    asample = Labsample[:, 1]
    bsample = Labsample[:, 2]
    Cabsample = np.sqrt(asample**2 + bsample**2)

    Cabarithmean = (Cabstd + Cabsample) / 2.0

    G = 0.5 * (1 - np.sqrt((Cabarithmean**7) / (Cabarithmean**7 + 25**7)))

    apstd = (1 + G) * astd
    apsample = (1 + G) * asample
    Cpsample = np.sqrt(apsample**2 + bsample**2)
    Cpstd = np.sqrt(apstd**2 + bstd**2)

    Cpprod = Cpsample * Cpstd
    zcidx = (Cpprod == 0)

    # Hue in radians
    hpstd = np.arctan2(bstd, apstd)
    hpstd[hpstd < 0] += 2 * np.pi
    hpstd[(np.abs(apstd) + np.abs(bstd)) == 0] = 0

    hpsample = np.arctan2(bsample, apsample)
    hpsample[hpsample < 0] += 2 * np.pi
    hpsample[(np.abs(apsample) + np.abs(bsample)) == 0] = 0

    dL = Lsample - Lstd
    dC = Cpsample - Cpstd

    dhp = hpsample - hpstd
    dhp[dhp > np.pi] -= 2 * np.pi
    dhp[dhp < -np.pi] += 2 * np.pi
    dhp[zcidx] = 0

    dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2.0)

    Lp = (Lsample + Lstd) / 2.0
    Cp = (Cpstd + Cpsample) / 2.0

    hp = (hpstd + hpsample) / 2.0
    hp[np.abs(hpstd - hpsample) > np.pi] -= np.pi
    hp[hp < 0] += 2 * np.pi
    hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]

    Lpm502 = (Lp - 50)**2
    Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
    Sc = 1 + 0.045 * Cp
    T = (1 - 0.17 * np.cos(hp - np.pi / 6.0) + 
         0.24 * np.cos(2 * hp) + 
         0.32 * np.cos(3 * hp + np.pi / 30.0) - 
         0.20 * np.cos(4 * hp - 63 * np.pi / 180.0))
    Sh = 1 + 0.015 * Cp * T
    
    delthetarad = (30 * np.pi / 180.0) * np.exp(-((180.0 / np.pi * hp - 275) / 25.0)**2)
    Rc = 2 * np.sqrt((Cp**7) / (Cp**7 + 25**7))
    RT = -np.sin(2 * delthetarad) * Rc

    klSl = kl * Sl
    kcSc = kc * Sc
    khSh = kh * Sh

    de00 = np.sqrt((dL / klSl)**2 + (dC / kcSc)**2 + (dH / khSh)**2 + RT * (dC / kcSc) * (dH / khSh))

    if is_1d:
        return float(de00[0])
    return de00

def generate_ground_truth_achrom():
    N = 20
    Yo = np.linspace(1, 160, 5)
    Y = np.linspace(0.1, 120, N)

    To = np.column_stack((Yo, Yo, Yo))
    T = np.column_stack((Y, Y, Y))

    lab1 = xyz2lab(T, To[0, :])
    lab2 = xyz2lab(T, To[1, :])
    lab3 = xyz2lab(T, To[2, :])
    lab4 = xyz2lab(T, To[3, :])
    lab5 = xyz2lab(T, To[4, :])

    delta_1 = deltaE2000(lab1[0, :], lab1)
    delta_2 = deltaE2000(lab2[0, :], lab2)
    delta_3 = deltaE2000(lab3[0, :], lab3)
    delta_4 = deltaE2000(lab4[0, :], lab4)
    delta_5 = deltaE2000(lab5[0, :], lab5)

    achrom = np.stack([delta_1, delta_2, delta_3, delta_4, delta_5])
    return achrom, Yo, Y, To, T

def generate_ground_truth_chrom():
    # Load colors_sat_nonlin
    data_path = os.path.dirname(__file__)
    data_path = os.path.join(data_path, 'colors_sat_nonlin.mat')
    mat = sio.loadmat(data_path)
    
    colors_A = mat['colors_A']
    colors_T = mat['colors_T']
    colors_D = mat['colors_D']
    
    C_back2 = mat['C_back2'].squeeze()
    C_back3m5 = mat['C_back3m5'].squeeze()
    C_back3m2 = mat['C_back3m2'].squeeze()
    C_back3M2 = mat['C_back3M2'].squeeze()
    C_back3M5 = mat['C_back3M5'].squeeze()
    C_back_T = np.stack([C_back3m5, C_back3m2, C_back2, C_back3M2, C_back3M5])
    
    C_back3m5D = mat['C_back3m5D'].squeeze()
    C_back3m2D = mat['C_back3m2D'].squeeze()
    C_back3M2D = mat['C_back3M2D'].squeeze()
    C_back3M5D = mat['C_back3M5D'].squeeze()
    C_back_D = np.stack([C_back3m5D, C_back3m2D, C_back2, C_back3M2D, C_back3M5D])

    # Scaling
    colors_T = (70/60) * colors_T
    colors_D = (70/60) * colors_D

    ## COLORS IN CIELAB WITH THE PROPER REFERENCE
    # Transposing because MATLAB uses 3x15, xyz2lab expects Nx3
    colorsTlabm5 = xyz2lab(colors_T.T, C_back3m5.T)
    colorsTlabm2 = xyz2lab(colors_T.T, C_back3m2.T)
    colorsTlab0 = xyz2lab(colors_T.T, C_back2.T)
    colorsTlabM2 = xyz2lab(colors_T.T, C_back3M2.T)
    colorsTlabM5 = xyz2lab(colors_T.T, C_back3M5.T)

    colorsDlabm5 = xyz2lab(colors_D.T, C_back3m5D.T)
    colorsDlabm2 = xyz2lab(colors_D.T, C_back3m2D.T)
    colorsDlab0 = xyz2lab(colors_D.T, C_back2.T)
    colorsDlabM2 = xyz2lab(colors_D.T, C_back3M2D.T)
    colorsDlabM5 = xyz2lab(colors_D.T, C_back3M5D.T)

    ## DIFFERENCES IN DELTA E
    ref_white = np.array([100, 0, 0])

    def calc_delta_e_with_sign(lab_vals, sign_col_idx):
        de = deltaE2000(ref_white, lab_vals)
        return np.sign(lab_vals[:, sign_col_idx]) * de

    delta_Tm5 = calc_delta_e_with_sign(colorsTlabm5, 1)
    delta_Tm2 = calc_delta_e_with_sign(colorsTlabm2, 1)
    delta_T0 = calc_delta_e_with_sign(colorsTlab0, 1)
    delta_TM2 = calc_delta_e_with_sign(colorsTlabM2, 1)
    delta_TM5 = calc_delta_e_with_sign(colorsTlabM5, 1)
    delta_T = np.stack([delta_Tm5, delta_Tm2, delta_T0, delta_TM2, delta_TM5])

    delta_Dm5 = calc_delta_e_with_sign(colorsDlabm5, 2)
    delta_Dm2 = calc_delta_e_with_sign(colorsDlabm2, 2)
    delta_D0 = calc_delta_e_with_sign(colorsDlab0, 2)
    delta_DM2 = calc_delta_e_with_sign(colorsDlabM2, 2)
    delta_DM5 = calc_delta_e_with_sign(colorsDlabM5, 2)
    delta_D = np.stack([delta_Dm5, delta_Dm2, delta_D0, delta_DM2, delta_DM5])

    x_axis = np.linspace(-20, 20, 15)

    return colors_T.T, colors_D.T, C_back_T, C_back_D, delta_T, delta_D, x_axis

def get_ground_truth():

    achrom, bgs, lum, t0, t = generate_ground_truth_achrom()
    colors_t, colors_d, bg_t, bg_d, delta_t, delta_d, x = generate_ground_truth_chrom()
    return (achrom, bgs, lum, t0, t), (colors_t, colors_d, bg_t, bg_d, delta_t, delta_d, x)

def xyz2ng(img):
    return np.power(img @ Mxyz2ng.T, gamma)

def generate_data(img_size,
                  square_size,
                  ):
    h, w = img_size
    hs, ws = square_size

    achrom, _, _, bg_a, colors_a = generate_ground_truth_achrom()

    ## Achrom
    achrom_bg = np.empty(shape=(len(bg_a), *img_size,3))
    for i, bg in enumerate(bg_a):
        achrom_bg[i] = bg

    achrom = np.repeat(achrom_bg[:,None], len(colors_a), axis=1)

    for i, bg_ in enumerate(achrom):
        for j, bg in enumerate(bg_):
            bg[h//2-hs//2:h//2+hs//2,
               w//2-ws//2:w//2+ws//2] = colors_a[j]

    achrom = xyz2ng(achrom)

    colors_t, colors_d, bg_t, bg_d, delta_t, delta_d, x = generate_ground_truth_chrom()
    ## Red-Green
    rg_bg = np.empty(shape=(len(bg_t), *img_size,3))
    for i, bg in enumerate(bg_t):
        rg_bg[i] = bg

    rg = np.repeat(rg_bg[:,None], len(colors_t), axis=1)

    for i, bg_ in enumerate(rg):
        for j, bg in enumerate(bg_):
            bg[h//2-hs//2:h//2+hs//2,
               w//2-ws//2:w//2+ws//2] = colors_t[j]

    rg = xyz2ng(rg)

    ## The reference image must contain the square as well
    for bg, idx in zip(rg_bg, [2,5,7,9,12]):
            bg[h//2-hs//2:h//2+hs//2,
               w//2-ws//2:w//2+ws//2] = colors_t[idx]

    ## Yellow-Blue
    yb_bg = np.empty(shape=(len(bg_d), *img_size,3))
    for i, bg in enumerate(bg_d):
        yb_bg[i] = bg

    yb = np.repeat(yb_bg[:,None], len(colors_d), axis=1)

    for i, bg_ in enumerate(yb):
        for j, bg in enumerate(bg_):
            bg[h//2-hs//2:h//2+hs//2,
               w//2-ws//2:w//2+ws//2] = colors_d[j]

    yb = xyz2ng(yb)
    
    ## The reference image must contain the square as well
    for bg, idx in zip(yb_bg, [2,5,7,9,12]):
            bg[h//2-hs//2:h//2+hs//2,
               w//2-ws//2:w//2+ws//2] = colors_d[idx]

    return (achrom, rg, yb), (xyz2ng(achrom_bg), xyz2ng(rg_bg), xyz2ng(yb_bg))

def evaluate_gen(calculate_diffs,
                 img_size: Sequence[int],
                 square_size: Sequence[int],
                 return_stimuli: bool = False,
                 return_gt: bool = False,
                 ):


    ## Generate data
    (data_a, data_rg, data_yb), (bg_a, bg_rg, bg_yb) = generate_data(img_size, square_size)
    stimuli = {"achrom": data_a, "red-green": data_rg, "yellow-blue": data_yb}
    plains = {"achrom": bg_a, "red-green": bg_rg, "yellow-blue": bg_yb}
    idxs_chrom = [2, 5, 7, 9, 12]

    results = {}
    for (name, stimuli_), (_, plain) in zip(stimuli.items(), plains.items()):
        diffs = np.empty(shape=stimuli_.shape[:2])
        for i, stims in enumerate(stimuli_):
            # print(f"Plain: {plain.shape}")
            # fig, axes = plt.subplots(1,5)
            # for p, ax in zip(plain, axes.ravel()):
            #     ax.imshow(p)
            # plt.show()
            # fig, axes = plt.subplots(1, len(stims))
            # for ax, s in zip(axes.ravel(), stims):
            #     ax.imshow(s)
            #     ax.axis("off")
            # plt.show()
            # plt.imshow(plain[i])
            # plt.show()
            if name != "achrom":
                diff = calculate_diffs(stims, plain[i:i+1])
                mask = np.ones_like(diff)
                mask[:idxs_chrom[i]] *= -1
                diff = mask*diff
            else:
                diff = calculate_diffs(stims, stims[0:1])
            diffs[i] = diff

        results[name] = diffs

    ## Generate ground truth
    (achrom, bgs, lum, t0, t), (colors_t, colors_d, bg_t, bg_d, delta_t, delta_d, x) = get_ground_truth()
    gt = {"achrom": achrom, "red-green": delta_t, "yellow-blue": delta_t}

    ## Correlations have to be calculated all together
    correlations = {}
    preds = np.concatenate([a.ravel() for a in results.values()])
    gts = np.concatenate([a.ravel() for a in gt.values()])
    correlations["pearson"] = pearsonr(preds, gts)[0]


    if return_stimuli and return_gt:
        return results, stimuli, correlations, gt
    elif return_stimuli and not return_gt:
        return results, stimuli, correlations
    if not return_stimuli and return_gt:
        return results, correlations, gt

    return results, correlations

