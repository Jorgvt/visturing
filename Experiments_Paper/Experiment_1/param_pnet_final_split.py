#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import cv2


from visturing.properties.prop1 import load_data, load_ground_truth
from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth, calculate_correlations, prepare_and_correlate


root_path = "../../Data/Experiment_1"
ref_path = os.path.join(root_path, "im_ref.png")


imgs_path = [p for p in glob(os.path.join(root_path, "*png")) if "ref" not in p]
imgs_path = list(natsorted(imgs_path))


def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
    return img


ref_img = load_img(ref_path)
imgs = np.array([load_img(p) for p in imgs_path])

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


diffs_a = calculate_diffs(imgs, ref_img[None,...])

def calculate_diffs(img1, img2):
    output_a = model.apply({"params": params, **state}, img1, train=False)[...,64:96]
    output_b = model.apply({"params": params, **state}, img2, train=False)[...,64:96]
    return ((output_a - output_b)**2).mean(axis=(1,2,3))**(1/2)


diffs_rg = calculate_diffs(imgs, ref_img[None,...])


def calculate_diffs(img1, img2):
    output_a = model.apply({"params": params, **state}, img1, train=False)[...,96:]
    output_b = model.apply({"params": params, **state}, img2, train=False)[...,96:]
    return ((output_a - output_b)**2).mean(axis=(1,2,3))**(1/2)


diffs_yb = calculate_diffs(imgs, ref_img[None,...])

_, _, x_e = load_data("../../Data/Experiment_1")

x_gt, y_a_gt, y_rg_gt, y_yb_gt = load_ground_truth("../../ground_truth_decalogo")

diffs_s = np.stack([diffs_a, diffs_rg, diffs_yb])
gt_s = np.stack([y_a_gt, y_rg_gt, y_yb_gt])


bs, ds = [], []
for b, d in zip(diffs_s, gt_s):
    a, b, c, d = prepare_data(x_e, b, x_gt, d)
    bs.append(b)
    ds.append(d)
b = np.array(bs)
d = np.array(ds)

print(f"Correlation (Order): {calculate_correlations_with_ground_truth(b, d)}")

import scipy.stats as stats
print(f"Pearson (Global): {stats.pearsonr(b.ravel(), d.ravel())}")
