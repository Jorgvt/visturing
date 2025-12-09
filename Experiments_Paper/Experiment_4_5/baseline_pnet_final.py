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

# bg = {p.split("/")[-1].split(".")[0].split("_")[-1]:np.load(p) for p in glob(os.path.join(root_path, "*")) if "background" in p}
bg = np.load(os.path.join(root_path, "background.npy"))


import json
import argparse

import jax
from jax import random, numpy as jnp
import flax
from huggingface_hub import hf_hub_download
from ml_collections import ConfigDict

from paramperceptnet.models import Baseline as PerceptNet
from paramperceptnet.configs import param_config

parser = argparse.ArgumentParser(description="Run parameter perception network experiment.")
parser.add_argument("--model-name", type=str, default="ppnet-baseline",
                    help="Name of the model to use (e.g., ppnet-bio-fitted)")
args = parser.parse_args()

model_name = args.model_name


config_path = hf_hub_download(repo_id=f"Jorgvt/{model_name}",
                              filename="config.json")
with open(config_path, "r") as f:
    config = ConfigDict(json.load(f))


from safetensors.flax import load_file

weights_path = hf_hub_download(repo_id=f"Jorgvt/{model_name}",
                               filename="weights.safetensors")
variables = load_file(weights_path)
variables = flax.traverse_util.unflatten_dict(variables, sep=".")
params = variables["params"]


model = PerceptNet(config)


def calculate_diffs(img1, img2):
    output_a = model.apply({"params": params}, img1, train=False)
    output_b = model.apply({"params": params}, img2, train=False)
    return ((output_a - output_b)**2).mean(axis=(1,2,3))**(1/2)


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


freqs = load_data("../..//Data/Experiment_4_5")


bs, ds = [], []
for d, gt in zip(diffs_s, gt_s):
    a, b, c, d = prepare_data(freqs, d, x_gt, gt)
    bs.append(b)
    ds.append(d)
b = np.array(bs)
d = np.array(ds)

print(f"Correlation (Order): {calculate_correlations_with_ground_truth(b, d)}")
import scipy.stats as stats


print(f"Pearson : {stats.pearsonr(b.ravel(), d.ravel())}")
