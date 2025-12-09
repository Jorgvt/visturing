#!/usr/bin/env python
# coding: utf-8

import os
import re
from glob import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


root_path = "../../Data/Experiment_6_7"


data = {re.findall("noise_(\w+)\.", p)[0]: np.load(p) for p in glob(os.path.join(root_path, "*"))}
for k, v in data.items(): print(k, v.shape)


freqs = np.array([1.5, 3, 6, 12, 24])
freqs


# for name, chroma in data.items():
#     for dat in chroma:
#         fig, axes = plt.subplots(1,len(dat))
#         for ax, d in zip(axes.ravel(), dat):
#             ax.imshow(d)
#             ax.axis("off")
#         plt.show()


def model(img):
    return img
def calculate_diffs(img1, img2):
    a, b = model(img1), model(img2)
    return ((a-b)**2).mean(axis=(1,2,3))**(1/2)


diffs = defaultdict(dict)
for name, chroma in data.items():
    for f, dat in zip(freqs, chroma):
        diffs_ = calculate_diffs(dat, dat[0:1])
        diffs[name][f] = diffs_


fig, axes = plt.subplots(1,len(diffs), figsize=(12,4))
for (name, chroma), ax in zip(diffs.items(), axes.ravel()):
    for f, dat in chroma.items():
        ax.plot(dat, label=f)
    ax.legend()
    ax.set_title(name)
plt.show()

