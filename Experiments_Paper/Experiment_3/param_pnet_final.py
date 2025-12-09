#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt


from visturing.properties.prop5 import load_ground_truth
from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth, calculate_correlations, prepare_and_correlate, prepare_and_correlate_order, calculate_spearman


x_gt, y1_gt, y2_gt, y3_gt = load_ground_truth("../../ground_truth_decalogo")
x_gt.shape, y1_gt.shape, y2_gt.shape, y3_gt.shape


root_path = "../../Data/Experiment_3"


noises = {p.split("/")[-1].split(".")[0].split("_")[-1]: np.load(p) for p in glob(os.path.join(root_path, "*npy")) if "noises" in p}

bgs = {p.split("/")[-1].split(".")[0].split("_")[-1]: np.load(p) for p in glob(os.path.join(root_path, "*npy")) if "background" in p}

freqs = np.load(os.path.join(root_path, "freqs.npy"))
freqs


import json
import argparse

import jax
from jax import random, numpy as jnp
import flax
from huggingface_hub import hf_hub_download
from ml_collections import ConfigDict

from paramperceptnet.models import PerceptNet
from paramperceptnet.configs import param_config

parser = argparse.ArgumentParser(description="Run parameter perception network experiment.")
parser.add_argument("--model-name", type=str, default="ppnet-bio-fitted",
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
state = variables["state"]
params = variables["params"]


model = PerceptNet(config)


@jax.jit
def calculate_diffs(img1, img2):
    output_a = model.apply({"params": params, **state}, img1, train=False)
    output_b = model.apply({"params": params, **state}, img2, train=False)
    return ((output_a - output_b)**2).mean(axis=(1,2,3))**(1/2)


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

a, b, c, d1 = prepare_data(freqs[1:], diffs_inv[k][1:], x_gt, y1_gt)
a, b, c, d2 = prepare_data(freqs[1:], diffs_inv[k][1:], x_gt, y2_gt)
a, b, c, d3 = prepare_data(freqs[1:], diffs_inv[k][1:], x_gt, y3_gt)


diffs_stack = np.stack([diffs_inv["3"][1:],
                        diffs_inv["6"][1:],
                        diffs_inv["12"][1:]])
ds = np.stack([d1, d2, d3])

print(f"Correlation Order: {calculate_correlations_with_ground_truth(diffs_stack, ds)}")


import scipy.stats as stats


print(f"Pearson: {stats.pearsonr(diffs_stack.ravel(), ds.ravel())}")
