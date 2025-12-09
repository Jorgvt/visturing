#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt


root_path = "../../Data/Experiment_4_5"


noises = {p.split("/")[-1].split(".")[0].split("_")[-1]:np.load(p) for p in glob(os.path.join(root_path, "*")) if "noises" in p}
for k,v in noises.items(): print(k, v.shape)


# bg = {p.split("/")[-1].split(".")[0].split("_")[-1]:np.load(p) for p in glob(os.path.join(root_path, "*")) if "background" in p}
bg = np.load(os.path.join(root_path, "background.npy"))
bg.shape


def model(img):
    return img
def calculate_diffs(img1, img2):
    a, b = model(img1), model(img2)
    return ((a-b)**2).mean(axis=(1,2,3))**(1/2)


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
    # break


diffs


fig, ax = plt.subplots()
for k, v in diffs.items():
    if k == "a": color = "k"
    elif k == "rg": color = "red"
    elif k == "yb": color = "blue"
    ax.plot(v, label=k, color=color)
plt.show()

