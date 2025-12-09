#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt


from visturing.properties.prop3_4 import load_data, load_ground_truth
from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth, calculate_correlations, prepare_and_correlate, prepare_and_correlate_order, calculate_spearman


x_gt, y_gt, rg_gt, yb_gt = load_ground_truth("../../ground_truth_decalogo")


root_path = "../../Data/Experiment_4_5"


noises = {p.split("/")[-1].split(".")[0].split("_")[-1]:np.load(p) for p in glob(os.path.join(root_path, "*")) if "noises" in p}
for k,v in noises.items(): print(k, v.shape)


for k, n in noises.items():
    print(k)
    fig, axes = plt.subplots(1,40, figsize=(40,20))
    for ax, im in zip(axes.ravel(), n[0]):
        ax.imshow(im)
        ax.axis("off")
    plt.show()


# bg = {p.split("/")[-1].split(".")[0].split("_")[-1]:np.load(p) for p in glob(os.path.join(root_path, "*")) if "background" in p}
bg = np.load(os.path.join(root_path, "background.npy"))
bg.shape


import json

import jax
from jax import random, numpy as jnp
import flax
from huggingface_hub import hf_hub_download
from ml_collections import ConfigDict

from paramperceptnet.models import PerceptNet
from paramperceptnet.configs import param_config


model_name = "ppnet-bio-fitted"


config_path = hf_hub_download(repo_id=f"Jorgvt/{model_name}",
                              filename="config.json")
with open(config_path, "r") as f:
    config = ConfigDict(json.load(f))


from safetensors.flax import load_file

weights_path = hf_hub_download(repo_id=f"Jorgvt/{model_name}",
                               filename="weights.safetensors")
variables = load_file(weights_path)
variables = flax.traverse_util.unflatten_dict(variables, sep=".")
state = variables["state"]
params = variables["params"]


# params["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]


model = PerceptNet(config)


params.keys()


layer = "GDNSpatioChromaFreqOrient_0"
# layer = "GDNGaussian_0"


def calculate_diffs(img1, img2):
    _, extra_a = model.apply({"params": params, **state}, img1, train=False, capture_intermediates=True)
    _, extra_b = model.apply({"params": params, **state}, img2, train=False, capture_intermediates=True)
    a = extra_a["intermediates"][layer]["__call__"][0]#[...,0:64]
    b = extra_b["intermediates"][layer]["__call__"][0]#[...,0:64]

    return ((a-b)**2).mean(axis=(1,2,3))**(1/2)


diffs = {}
k, noises_ = "a", noises["a"]
diffs_it = []
for noise_it in noises_:
    diff = calculate_diffs(noise_it, bg[None,...])
    # print(noise_it.shape, bg.shape, diff.shape)
    diffs_it.append(diff)
    # break
diffs_it = np.array(diffs_it)
diffs[k] = diffs_it.mean(axis=0)
# break


def calculate_diffs(img1, img2):
    _, extra_a = model.apply({"params": params, **state}, img1, train=False, capture_intermediates=True)
    _, extra_b = model.apply({"params": params, **state}, img2, train=False, capture_intermediates=True)
    # a = extra_a["intermediates"][layer]["__call__"][0][...,1:2]
    # b = extra_b["intermediates"][layer]["__call__"][0][...,1:2]
    a = extra_a["intermediates"][layer]["__call__"][0]#[...,64:96]
    b = extra_b["intermediates"][layer]["__call__"][0]#[...,64:96]
    return ((a-b)**2).mean(axis=(1,2,3))**(1/2)


# diffs = {}
k, noises_ = "rg", noises["rg"]
diffs_it = []
for noise_it in noises_:
    diff = calculate_diffs(noise_it, bg[None,...])
    # print(noise_it.shape, bg.shape, diff.shape)
    diffs_it.append(diff)
    # break
diffs_it = np.array(diffs_it)
diffs[k] = diffs_it.mean(axis=0)
# break


def calculate_diffs(img1, img2):
    _, extra_a = model.apply({"params": params, **state}, img1, train=False, capture_intermediates=True)
    _, extra_b = model.apply({"params": params, **state}, img2, train=False, capture_intermediates=True)
    # a = extra_a["intermediates"][layer]["__call__"][0][...,2:3]
    # b = extra_b["intermediates"][layer]["__call__"][0][...,2:3]
    a = extra_a["intermediates"][layer]["__call__"][0]#[...,96:]
    b = extra_b["intermediates"][layer]["__call__"][0]#[...,96:]
    return ((a-b)**2).mean(axis=(1,2,3))**(1/2)


# diffs = {}
k, noises_ = "yb", noises["yb"]
diffs_it = []
for noise_it in noises_:
    diff = calculate_diffs(noise_it, bg[None,...])
    # print(noise_it.shape, bg.shape, diff.shape)
    diffs_it.append(diff)
    # break
diffs_it = np.array(diffs_it)
diffs[k] = diffs_it.mean(axis=0)
# break


fig, ax = plt.subplots()
for k, v in diffs.items():
    if k == "a": color = "k"
    elif k == "rg": color = "red"
    elif k == "yb": color = "blue"
    ax.plot(v, label=k, color=color)
plt.xscale("log")
plt.yscale("log")
# plt.ylim([1e-3, 0.1])
plt.legend()
plt.show()


gt_s = np.stack([y_gt,
                 rg_gt,
                 yb_gt])
gt_s.shape


diffs_s = np.stack([diffs["a"],
                    diffs["rg"],
                    diffs["yb"]])
diffs_s.shape


freqs = load_data("../..//Data/Experiment_4_5")
freqs.shape


bs, ds = [], []
for d, gt in zip(diffs_s, gt_s):
    a, b, c, d = prepare_data(freqs, d, x_gt, gt)
    bs.append(b)
    ds.append(d)
b = np.array(bs)
d = np.array(ds)
a.shape, b.shape, c.shape, d.shape


calculate_correlations_with_ground_truth(b, d)


import scipy.stats as stats


stats.pearsonr(b.ravel(), d.ravel())

