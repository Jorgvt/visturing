import jax
from jax import random, numpy as jnp
import flax.linen as nn

from visturing.properties.jax import prop5


def test_differentiability():
    def loss_fn(params):
        def calculate_diffs(img1, img2):
            return params*((img1-img2).mean(axis=(1,2,3))**2)**(1/2)
        output = prop5.evaluate(calculate_diffs,
                                            data_path='../../../Data/Experiment_5',
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
            output = prop5.evaluate(calculate_diffs,
                                                data_path='../../../Data/Experiment_5',
                                                gt_path='../../../Data/ground_truth/')
            return output["correlations"]["pearson"]

        loss, grad = jax.value_and_grad(loss_fn)(params)
        params = params - 0.01 * grad
        return params, loss, grad

    params = jnp.array([1.])
    params, loss, grad = jax.jit(step)(params)

    assert not jnp.isnan(params)
    assert not jnp.isnan(loss)
    assert not jnp.isnan(grad)


def test_jit_differentiability_simple():
    def step(params):
        def loss_fn(params):
            def calculate_diffs(img1, img2):
                p1 = model.apply({"params": params}, img1)
                p2 = model.apply({"params": params}, img2)
                return ((p1-p2).mean(axis=(1,2,3))**2+1e-6)**(1/2)
            output = prop5.evaluate(calculate_diffs,
                                                data_path='../../../Data/Experiment_5',
                                                gt_path='../../../Data/ground_truth/')
            return output["correlations"]["pearson"]

        loss, grad = jax.value_and_grad(loss_fn)(params)
        params = jax.tree_util.tree_map(lambda x,g: x- 0.01 * g, params, grad)
        return params, loss, grad

    model = nn.Conv(1, (3,3))
    variables = model.init(random.PRNGKey(0), jnp.ones((1,256,256,3)))
    params = variables['params']
    params, loss, grad = jax.jit(step)(params)

    assert not jnp.isnan(loss)
    assert not jnp.isnan(
                jnp.array(
                    jnp.concatenate(
                        [a.ravel() for a in jax.tree_util.tree_leaves(grad)]
                    )
                )
            ).all()
    assert not jnp.isnan(
                jnp.array(
                    jnp.concatenate(
                        [a.ravel() for a in jax.tree_util.tree_leaves(params)]
                    )
                )
            ).all()
