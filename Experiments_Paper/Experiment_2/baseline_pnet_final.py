#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt


from visturing.properties.prop2 import load_data, load_ground_truth
from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth, calculate_correlations, prepare_and_correlate, prepare_and_correlate_order, calculate_spearman


root_path = "../../Data/Experiment_2"


x, y, x_c, rg, x_c, yb = load_ground_truth("../../ground_truth_decalogo")


x_a, x_rg, x_yb = load_data("../../Data/Experiment_2")


data = {p.split("/")[-1].split(".")[0]: np.load(p) for p in glob(os.path.join(root_path, "*npy")) if "bgs" not in p}

bgs = {p.split("/")[-1].split(".")[0][4:]: np.load(p) for p in glob(os.path.join(root_path, "*npy")) if "bgs" in p}

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
for c in ["achrom", "red_green", "yellow_blue"]:
    diffs[c] = []
    data_ = data[c]
    bgs_ = bgs[c]
    for cc, bg in zip(data_, bgs_):
        diff = calculate_diffs(cc, bg[None,...])
        diffs[c].append(diff)
diffs = {k: np.array(v) for k, v in diffs.items()}

a, b, c, d = prepare_data(x_a, diffs["achrom"], x, y)
print(f"Order Correlation (Achrom): {calculate_spearman(b, ideal_ordering=[0,1,2,3,4])}")


a, b_rg, c, d_rg = prepare_data(x_rg, diffs["red_green"], x_c, rg)
print(f"Order Correlation (RG): {calculate_spearman(b_rg, ideal_ordering=[0,1,2,3,4])}")


a, b_yb, c, d_yb = prepare_data(x_yb, diffs["yellow_blue"], x_c, yb)
print(f"Order Correlation (YB): {calculate_spearman(b_yb, ideal_ordering=[0,1,2,3,4])}")


import scipy.stats as stats


corr_achrom = stats.pearsonr(
    np.concatenate([
        b[0].ravel(),
    ]),
    np.concatenate([
        d.ravel(),
    ])
)

print(f"Pearson (Achromatic): {corr_achrom}")

corr_chroma = stats.pearsonr(
    np.concatenate([
        b_rg[2].ravel(), b_yb[2].ravel(),
    ]),
    np.concatenate([
        d_rg.ravel(), d_yb.ravel(),
    ])
)
print(f"Pearson (Chromatic): {corr_chroma}")
