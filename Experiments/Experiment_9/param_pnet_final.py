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


root_path = "../../Data/Experiment_9"


import scipy
exp8_low = scipy.io.loadmat('resp_legge_energy_3.mat')
exp8_high = scipy.io.loadmat('resp_legge_energy_12.mat')

xs = exp8_high["C"][0]


data_high = {re.findall("high_(\w+)\.", p)[0]: np.load(p) for p in glob(os.path.join(root_path, "*")) if "high" in p}


data_low = {re.findall("low_(\w+)\.", p)[0]: np.load(p) for p in glob(os.path.join(root_path, "*")) if "_low_" in p}


f_mask = ['No mask', 'F_mask = 1.5 cpd', 'F_mask = 3 cpd', 'F_mask = 6 cpd', 'F_mask = 12 cpd', 'F_mask = 24 cpd']


# for name, chroma in data.items():
#     for dat in chroma:
#         fig, axes = plt.subplots(1,len(dat))
#         for ax, d in zip(axes.ravel(), dat):
#             ax.imshow(d)
#             ax.axis("off")
#         plt.show()


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

def calculate_diffs(img1, img2):
    output_a = model.apply({"params": params, **state}, img1, train=False)
    output_b = model.apply({"params": params, **state}, img2, train=False)
    return ((output_a - output_b)**2).mean(axis=(1,2,3))**(1/2)


diffs_high = defaultdict(dict)
for name, chroma in data_high.items():
    for f, dat in zip(f_mask, chroma):
        diffs_ = calculate_diffs(dat, dat[0:1])
        diffs_high[name][f] = diffs_

diffs_low = defaultdict(dict)
for name, chroma in data_low.items():
    for f, dat in zip(f_mask, chroma):
        diffs_ = calculate_diffs(dat, dat[0:1])
        diffs_low[name][f] = diffs_


diffs_low_s = np.array([a for a in diffs_low["achrom"].values()])

print(f"Correlations (Order) [Low]: {calculate_spearman(diffs_low_s, ideal_ordering=[0,4,5,3,2,1])}")

diffs_high_s = np.array([a for a in diffs_high["achrom"].values()])

print(f"Correlations (Order) [Low]: {calculate_spearman(diffs_high_s, ideal_ordering=[0,1,2,3,5,4])}")
