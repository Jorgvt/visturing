#!/usr/bin/env python
# coding: utf-8

import os
import re
from glob import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


from visturing.properties.prop8 import load_ground_truth
from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth, calculate_correlations, prepare_and_correlate, prepare_and_correlate_order, calculate_spearman, calculate_pearson_stack


root_path = "../../Data/Experiment_8"


import scipy
exp8_low = scipy.io.loadmat('resp_legge_energy_3.mat')
exp8_high = scipy.io.loadmat('resp_legge_energy_12.mat')

xs = exp8_high["C"][0]

data_high = {re.findall("high_(\w+)\.", p)[0]: np.load(p) for p in glob(os.path.join(root_path, "*")) if "high" in p}

data_low = {re.findall("low_(\w+)\.", p)[0]: np.load(p) for p in glob(os.path.join(root_path, "*")) if "_low_" in p}


c_mask = ['No mask', 'C_mask = 0.075', 'C_mask = 0.150', 'C_mask = 0.225', 'C_mask = 0.300']


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


model = PerceptNet(config)

def calculate_diffs(img1, img2):
    output_a = model.apply({"params": params, **state}, img1, train=False)[...,:64]
    output_b = model.apply({"params": params, **state}, img2, train=False)[...,:64]
    return ((output_a - output_b)**2).mean(axis=(1,2,3))**(1/2)


diffs_high = defaultdict(dict)
for name, chroma in data_high.items():
    if name != "achrom": continue
    for f, dat in zip(c_mask, chroma):
        diffs_ = calculate_diffs(dat, dat[0:1])
        diffs_high[name][f] = diffs_

def calculate_diffs(img1, img2):
    output_a = model.apply({"params": params, **state}, img1, train=False)[...,64:96]
    output_b = model.apply({"params": params, **state}, img2, train=False)[...,64:96]
    return ((output_a - output_b)**2).mean(axis=(1,2,3))**(1/2)

for name, chroma in data_high.items():
    if name != "red_green": continue
    for f, dat in zip(c_mask, chroma):
        diffs_ = calculate_diffs(dat, dat[0:1])
        diffs_high[name][f] = diffs_

def calculate_diffs(img1, img2):
    output_a = model.apply({"params": params, **state}, img1, train=False)[...,96:]
    output_b = model.apply({"params": params, **state}, img2, train=False)[...,96:]
    return ((output_a - output_b)**2).mean(axis=(1,2,3))**(1/2)

for name, chroma in data_high.items():
    if name != "yellow_blue": continue
    for f, dat in zip(c_mask, chroma):
        diffs_ = calculate_diffs(dat, dat[0:1])
        diffs_high[name][f] = diffs_

def calculate_diffs(img1, img2):
    output_a = model.apply({"params": params, **state}, img1, train=False)[...,:64]
    output_b = model.apply({"params": params, **state}, img2, train=False)[...,:64]
    return ((output_a - output_b)**2).mean(axis=(1,2,3))**(1/2)

diffs_low = defaultdict(dict)
for name, chroma in data_low.items():
    if name != "achrom": continue
    for f, dat in zip(c_mask, chroma):
        diffs_ = calculate_diffs(dat, dat[0:1])
        diffs_low[name][f] = diffs_

def calculate_diffs(img1, img2):
    output_a = model.apply({"params": params, **state}, img1, train=False)[...,64:96]
    output_b = model.apply({"params": params, **state}, img2, train=False)[...,64:96]
    return ((output_a - output_b)**2).mean(axis=(1,2,3))**(1/2)

for name, chroma in data_low.items():
    if name != "red_green": continue
    for f, dat in zip(c_mask, chroma):
        diffs_ = calculate_diffs(dat, dat[0:1])
        diffs_low[name][f] = diffs_

def calculate_diffs(img1, img2):
    output_a = model.apply({"params": params, **state}, img1, train=False)[...,96:]
    output_b = model.apply({"params": params, **state}, img2, train=False)[...,96:]
    return ((output_a - output_b)**2).mean(axis=(1,2,3))**(1/2)

for name, chroma in data_low.items():
    if name != "yellow_blue": continue
    for f, dat in zip(c_mask, chroma):
        diffs_ = calculate_diffs(dat, dat[0:1])
        diffs_low[name][f] = diffs_

x_gt, y_low_gt, y_high_gt  = load_ground_truth("../../ground_truth_decalogo")

diffs_low_a_s = np.array([a for a in diffs_low["achrom"].values()])
diffs_low_rg_s = np.array([a for a in diffs_low["red_green"].values()])
diffs_low_yb_s = np.array([a for a in diffs_low["yellow_blue"].values()])


bs = []
for b in diffs_low_a_s:
    a, b, c, d = prepare_data(xs, b, x_gt, y_low_gt)
    bs.append(b)
b_low = np.array(bs)

print(f"Correlation (Order) [Low]: {calculate_spearman(b_low, ideal_ordering=[0,1,2,3,4])}")


diffs_high_a_s = np.array([a for a in diffs_high["achrom"].values()])


bs = []
for b in diffs_high_a_s:
    a, b, c, d = prepare_data(xs, b, x_gt, y_high_gt)
    bs.append(b)
b_high = np.array(bs)

print(f"Correlation (Order) [High]: {calculate_spearman(b_high, ideal_ordering=[0,1,2,3,4])}")

import scipy.stats as stats
pears = stats.pearsonr(
    np.concatenate([
        b_low[0], b_high[0]
    ]),
    np.concatenate([
        y_low_gt, y_high_gt
    ])
)
print(f"Correlation (Pearson): {pears}")
