import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

from jax import random, numpy as jnp
from flax.core import pop
from model import Model
from initialization import init_dn_gamma, init_cs, init_dn_cs, init_v1, init_dn_v1
from visturing.properties import prop8 as prop

# Load the model
key = random.PRNGKey(42)
x = jnp.ones((1,128,128,3))
model = Model()
variables = model.init(key, x)
state, params = pop(variables, "params")
_, state = model.apply({"params": params, **state}, x, train=True, mutable=list(state.keys()))

params = init_dn_gamma(params)
params["CenterSurroundLogSigmaK_0"] = init_cs(params["CenterSurroundLogSigmaK_0"])
params, state = init_dn_cs(params, state)
params = init_v1(params)
params, state = init_dn_v1(params, state)

_, state = model.apply({"params": params, **state}, x, train=True, mutable=list(state.keys()))

# Config
img_size = (128, 128)
fs = 128
fm = 0.25
fM = 30
theta = 0
delta_theta = 30
sigma_mask = 0.3
R0 = 0.15
N = 15
L = 40

Cs = np.linspace(0.01, 0.2, num=10)
Cs_mask = np.linspace(0, 0.2, num=5)[1:]
Cs_mask = (((img_size[0]/fs)**2)/((2*R0)**2))**(1/2)*Cs_mask

c = 1
n_iters = 2

freqs = np.arange(1, 17, step=1)
freqs_mask = np.array([2, 4, 6, 8, 10])

# freqs = np.array([4.])
# freqs_mask = np.array([3.])

theta_mask = np.array([0.])

def calculate_diffs(a, b):
    batch, m, h, w, c = a.shape
    a = rearrange(a, "b bb h w c -> (b bb) h w c")
    # b = rearrange(b, "b bb h w c -> (b bb) h w c")

    a = model.apply({"params": params, **state}, a, train=False)
    b = model.apply({"params": params, **state}, b, train=False)

    a = rearrange(a, "(b bb) h w c -> b bb h w c", b=batch, bb=m)
    # b = rearrange(b, "(b bb) h w c -> b bb h w c", b=batch, bb=m)
    return ((a-b)**2).mean(axis=(-3,-2,-1))**(1/2)

results, freqs, stimuli, correlation = prop.evaluate_gen(
                calculate_diffs=calculate_diffs,
                img_size=img_size,
                freqs=freqs,
                freqs_mask=freqs_mask,
                L=L,
                Cs=Cs,
                Cs_mask=Cs_mask,
                fs=fs,
                sigma_mask=sigma_mask,
                n_iters=n_iters,
                theta=theta,
                theta_mask=theta_mask,
                delta_theta=delta_theta,
                return_stimuli=True,
                )

print(f"Correlation: {correlation}")
