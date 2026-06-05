import numpy as np
import matplotlib.pyplot as plt

from visturing.properties.noise import generate_noise, generate_noise_iters, generate_plain
from visturing.properties import prop3_4 as prop

from paramperceptnet.pretrained import load_param_pretrained

model, variables = load_param_pretrained()
params = variables["params"]
state = variables["state"]
inputs = np.ones(shape=(1,128,128,3))
pred, state = model.apply({"params": params, **state}, inputs, train=True, mutable=list(state.keys()))
print("Model loaded!")
