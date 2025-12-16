import jax
from jax import random, numpy as jnp

from visturing.properties.jax import prop6_7


def test_differentiability():
    def loss_fn(params):
        def calculate_diffs(img1, img2):
            return params*((img1-img2).mean(axis=(1,2,3))**2)**(1/2)
        output = prop6_7.evaluate(calculate_diffs,
                                            data_path='../../Data/Experiment_6_7',
                                            gt_path='../../Data/ground_truth/')
        return output["correlations"]["pearson"]

    params = jnp.array([1.])
    loss, grad = jax.value_and_grad(loss_fn)(params)
    assert not jnp.isnan(loss)
    assert not jnp.isnan(grad)
