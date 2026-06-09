import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from paramperceptnet.pretrained import load_param_pretrained
from visturing.properties import prop8
from visturing.properties.config import default_prop8_config

def main():
    print("Starting JAX PerceptNet optimization experiment on Prop. 8 using NON-WEIGHTED correlation...")

    model, variables = load_param_pretrained()
    params = variables["params"]
    state = variables["state"]

    dummy_x = jnp.zeros((1, 128, 128, 3))
    _, state = model.apply({"params": params, **state}, dummy_x, train=True, mutable=list(state.keys()))
    print("Precalculated filters populated in state!")

    tx = optax.adam(learning_rate=1e-4)
    opt_state = tx.init(params)

    def loss_fn(params_val):
        def calculate_diffs(a, b):
            a_j = jnp.asarray(a)
            b_j = jnp.asarray(b)
            # Use train=True and mutable to recalculate filters so gradients flow to all parameters
            feat_a, _ = model.apply({"params": params_val, **state}, a_j, train=True, mutable=list(state.keys()))
            feat_b, _ = model.apply({"params": params_val, **state}, b_j, train=True, mutable=list(state.keys()))
            diffs = jnp.sqrt(jnp.mean((feat_a - feat_b) ** 2, axis=(-3, -2, -1)) + 1e-8)
            return diffs

        res = prop8.evaluate_gen(
            calculate_diffs,
            xp=jnp,
            verbose=False,
            **default_prop8_config
        )
        
        corr = res.correlations["non-weighted"]["global"]
        loss = -corr
        return loss, corr

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    print("Initial evaluation...")
    loss, init_corr = loss_fn(params)
    print(f"Initial global non-weighted correlation: {init_corr:.4f}")

    print("\nRunning optimization loop (10 steps)...")
    for i in range(10):
        (loss_val, corr_val), grads = grad_fn(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        print(f"Step {i+1:02d} | Loss: {loss_val:.4f} | Correlation: {corr_val:.4f}")

    print("\nOptimization finished!")
    final_loss, final_corr = loss_fn(params)
    print(f"Final global non-weighted correlation: {final_corr:.4f}")
    print(f"Total improvement: {final_corr - init_corr:.4f}")

if __name__ == "__main__":
    main()
