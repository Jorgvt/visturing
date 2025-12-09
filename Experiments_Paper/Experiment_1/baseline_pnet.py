#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import cv2


from visturing.properties.prop1 import load_data, load_ground_truth
from visturing.ranking import prepare_data, calculate_correlations_with_ground_truth, calculate_correlations, prepare_and_correlate, calculate_spearman


root_path = "../../Data/Experiment_1"
ref_path = os.path.join(root_path, "im_ref.png")


imgs_path = [p for p in glob(os.path.join(root_path, "*png")) if "ref" not in p]
imgs_path = list(natsorted(imgs_path))
imgs_path[:6]


def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
    return img


ref_img = load_img(ref_path)
imgs = np.array([load_img(p) for p in imgs_path])
ref_img.shape, imgs.shape


fig, axes = plt.subplots(6,5)
for im, ax in zip(imgs, axes.ravel()):
    ax.imshow(im)
    ax.axis("off")
plt.show()


import json

import jax
from jax import random, numpy as jnp
import flax
from huggingface_hub import hf_hub_download
from ml_collections import ConfigDict

from paramperceptnet.models import Baseline as PerceptNet
from paramperceptnet.configs import param_config


model_name = "ppnet-baseline"


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
    _, extra_a = model.apply({"params": params}, img1, train=False, capture_intermediates=True)
    _, extra_b = model.apply({"params": params}, img2, train=False, capture_intermediates=True)
    a = extra_a["intermediates"]["Color"]["__call__"][0][...,0:1]
    b = extra_b["intermediates"]["Color"]["__call__"][0][...,0:1]
    return ((a-b)**2).mean(axis=(1,2,3))**(1/2)


diffs_a = calculate_diffs(imgs, ref_img[None,...])
diffs_a.shape


plt.plot(diffs_a)
plt.show()


_, _, x_e = load_data("../../Data/Experiment_1")


x_gt, y_a_gt, y_rg_gt, y_yb_gt = load_ground_truth("../../ground_truth_decalogo")


prepare_and_correlate(x_e, diffs, x_gt, y_a_gt)


def calculate_diffs(img1, img2):
    _, extra_a = model.apply({"params": params}, img1, train=False, capture_intermediates=True)
    _, extra_b = model.apply({"params": params}, img2, train=False, capture_intermediates=True)
    a = extra_a["intermediates"]["Color"]["__call__"][0][...,1:2]
    b = extra_b["intermediates"]["Color"]["__call__"][0][...,1:2]
    sign = jnp.sign(a-b).mean(axis=(1,2,3))
    return sign*((a-b)**2).mean(axis=(1,2,3))**(1/2)


diffs_rg = calculate_diffs(imgs, ref_img[None,...])
diffs_rg.shape


plt.plot(diffs_rg)
plt.show()


x_gt, y_a_gt, y_rg_gt, y_yb_gt = load_ground_truth("../../ground_truth_decalogo")


prepare_and_correlate(x_e, diffs, x_gt, y_rg_gt)


def calculate_diffs(img1, img2):
    _, extra_a = model.apply({"params": params}, img1, train=False, capture_intermediates=True)
    _, extra_b = model.apply({"params": params}, img2, train=False, capture_intermediates=True)
    a = extra_a["intermediates"]["Color"]["__call__"][0][...,2:3]
    b = extra_b["intermediates"]["Color"]["__call__"][0][...,2:3]
    sign = jnp.sign(a-b).mean(axis=(1,2,3))
    return sign*((a-b)**2).mean(axis=(1,2,3))**(1/2)


diffs_yb = calculate_diffs(imgs, ref_img[None,...])
diffs_yb.shape


plt.plot(diffs_yb)
plt.show()


x_gt, y_a_gt, y_rg_gt, y_yb_gt = load_ground_truth("../../ground_truth_decalogo")


prepare_and_correlate(x_e, diffs, x_gt, y_yb_gt)


diffs_s = np.stack([diffs_a, diffs_rg, diffs_yb])
diffs_s.shape


gt_s = np.stack([y_a_gt, y_rg_gt, y_yb_gt])
gt_s.shape


bs, ds = [], []
for b, d in zip(diffs_s, gt_s):
    a, b, c, d = prepare_data(x_e, b, x_gt, d)
    bs.append(b)
    ds.append(d)
b = np.array(bs)
d = np.array(ds)
b.shape, d.shape


import scipy.stats as stats
stats.pearsonr(b.ravel(), d.ravel())


calculate_correlations_with_ground_truth(b, d)

