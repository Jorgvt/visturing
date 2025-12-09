#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt


root_path = "../../Data/Experiment_3"


noises = {p.split("/")[-1].split(".")[0].split("_")[-1]: np.load(p) for p in glob(os.path.join(root_path, "*npy")) if "background" not in p}
for k,v in noises.items(): print(f"{k}: {v.shape}")


bgs = {p.split("/")[-1].split(".")[0].split("_")[-1]: np.load(p) for p in glob(os.path.join(root_path, "*npy")) if "background" in p}
for k,v in bgs.items(): print(f"{k}: {v.shape}")


freqs = np.load(os.path.join(root_path, "freqs.npy"))
freqs


def model(img):
    return img
def calculate_diffs(img1, img2):
    a, b = model(img1), model(img2)
    return ((a-b)**2).mean(axis=(1,2,3))**(1/2)


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
diffs_inv = {k:v/diffs_a for k, v in diffs.items()}


fig, axes = plt.subplots(1,len(diffs_inv))
for ax, (k, diff) in zip(axes.ravel(), diffs_inv.items()):
    ax.plot(freqs, diff)
    ax.set_title(k)
plt.show()


fig, ax = plt.subplots(1,1)
for k, diff in diffs_inv.items():
    if k == "3": style = "solid"
    elif k == "6": style = "dotted"
    elif k == "12": style = "dashed"
    ax.plot(freqs, diff, linestyle=style, color="k", label=f"{k} cpd")
    ax.set_title(k)
plt.legend()
plt.show()

