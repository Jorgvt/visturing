import jax
from jax import random, numpy as jnp

from visturing.properties.jax import prop8


def test_differentiability():
    def loss_fn(params):
        def calculate_diffs(img1, img2):
            return params*((img1-img2).mean(axis=(1,2,3))**2)**(1/2)
        output = prop8.evaluate(calculate_diffs,
                                            data_path='../../../Data/Experiment_8',
                                            gt_path='../../../Data/ground_truth/')
        return output["correlations"]["pearson"]

    params = jnp.array([1.])
    loss, grad = jax.value_and_grad(loss_fn)(params)
    assert not jnp.isnan(loss)
    assert not jnp.isnan(grad)


def test_jit_differentiability():
    def step(params):
        def loss_fn(params):
            def calculate_diffs(img1, img2):
                return params*((img1-img2).mean(axis=(1,2,3))**2)**(1/2)
            output = prop8.evaluate(calculate_diffs,
                                                data_path='../../../Data/Experiment_8',
                                                gt_path='../../../Data/ground_truth/')
            return output["correlations"]["pearson"]

        loss, grad = jax.value_and_grad(loss_fn)(params)
        params = params - 0.01 * grad
        return params, loss, grad

    params = jnp.array([1.])
    params, loss, grad = jax.jit(step)(params)

    # assert not jnp.isnan(params)
    # assert not jnp.isnan(loss)
    # assert not jnp.isnan(grad)
