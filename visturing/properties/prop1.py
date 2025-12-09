import os
import wget
from zipfile import ZipFile
from glob import glob

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from natsort import natsorted
import cv2
from scipy.stats import pearsonr

from .math_utils import pearson_correlation

def load_ground_truth(root_path: str = "ground_truth_decalogo", # Path to the root containing all the ground truth files
                      ): # Tuple (x, achromatic, red-green, yellow-blue)
    data = sio.loadmat(os.path.join(root_path, "spectral_sensitivities.mat"))
    data = data["spectral_sensitivities"]

    x = data[0]
    a = data[1]
    rg = data[2]
    yb = data[3]

    return (x, a, rg, yb)


def plot_ground_truth(x,
                      a,
                      rg,
                      yb,
                      ): # Returns both the fig and axes objects
    fig, axes = plt.subplots(1,2, sharex=True, sharey=False, figsize=(10,4))
    axes[0].plot(x, a, "k", label="Achromatic")

    axes[0].set_xlim([380, 720])
    axes[0].set_ylim([0, 1.1])
    axes[0].legend()

    axes[1].plot(x, rg, "r", label="Red-Green")
    axes[1].plot(x, yb, "b", label="Yellow-Blue")
    axes[1].set_xlim([380, 720])
    axes[1].set_ylim([-0.7, 0.7])

    axes[1].legend()

    axes[0].set_xlabel(r"Wavelength ($\lambda$)")
    axes[0].set_ylabel("Visibility")
    axes[1].set_xlabel(r"Wavelength ($\lambda$)")

    return fig, axes

def load_data(root_path: str,
              ): # Tuple (imgs, reference_image)
    ref_path = os.path.join(root_path, "im_ref.png")
    lambdas = np.load(os.path.join(root_path, "lambdas.npy"))

    imgs_path = [p for p in glob(os.path.join(root_path, "*png")) if "ref" not in p]
    imgs_path = list(natsorted(imgs_path))

    def load_img(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    ref_img = load_img(ref_path)
    imgs = np.array([load_img(p) for p in imgs_path])
    lambdas = np.linspace(lambdas.min(), lambdas.max(), num=len(imgs))

    return imgs, ref_img, lambdas


def evaluate(calculate_diffs,
             data_path: str = "Data/Experiment_1",
             gt_path: str = "ground_truth_decalogo",
             ): # Tuple (lambdas, diffs, correlation)

    if not os.path.exists(data_path):
        data_path = download_data("/".join(data_path.split("/")[:-1]))
    
    imgs, ref_img, lambdas = load_data(data_path)

    diffs = calculate_diffs(imgs, ref_img[None,...])

    x, a, _, _ = load_ground_truth(gt_path)
    a_interp = np.interp(lambdas, x, a)
    corr, p_value = pearsonr(diffs, a_interp)

    return {"lambdas": lambdas,
            "diffs": diffs,
            "correlations":
                {"pearson": corr},
            "p_values":
                {"pearson": p_value},
            }

def download_data(data_path, # Path to download the data
                  ):
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    data_url = "https://zenodo.org/records/17700252/files/Experiment_1.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(data_path, "Experiment_1")

