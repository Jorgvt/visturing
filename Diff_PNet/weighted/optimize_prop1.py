import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from paramperceptnet.pretrained import load_param_pretrained
from visturing.properties import prop1

def jax_pearson_correlation(x, y):
    mean_x = jnp.mean(x)
    mean_y = jnp.mean(y)
    dev_x = x - mean_x
    dev_y = y - mean_y
    cov_xy = jnp.sum(dev_x * dev_y)
    var_x = jnp.sum(dev_x ** 2)
    var_y = jnp.sum(dev_y ** 2)
    return cov_xy / jnp.sqrt(var_x * var_y + 1e-8)

def main():
    print("Starting JAX PerceptNet optimization experiment on Prop. 1 (Spectral Sensitivity)...")
    print("Note: Prop. 1 only has standard (non-weighted) human spectral sensitivities.")

    # Paths (relative to the repo root)
    data_path = "./Experiment_1"
    gt_path = "./ground_truth"

    if not os.path.exists(data_path):
        data_path = prop1.download_data(".")

    # Load data
    imgs, ref_img, lambdas = prop1.load_data(data_path)
    x, a, _, _ = prop1.load_ground_truth(gt_path)
    a_interp = np.interp(lambdas, x, a)
    a_interp_j = jnp.asarray(a_interp)

    print(f"Loaded {len(imgs)} stimuli images of shape {imgs.shape[1:]}")

    # Load pretrained PerceptNet model
    model, variables = load_param_pretrained()
    params = variables["params"]
    state = variables["state"]

    # Run a dummy forward pass to populate precalc_filter variables in state
    dummy_x = jnp.zeros((1, *imgs.shape[1:]))
    _, state = model.apply({"params": params, **state}, dummy_x, train=True, mutable=list(state.keys()))
    print("Precalculated filters populated in state!")

    # Initialize Optax optimizer (Adam)
    tx = optax.adam(learning_rate=1e-4)
    opt_state = tx.init(params)

    # Define loss function
    def loss_fn(params_val):
        a_j = jnp.asarray(imgs)
        b_j = jnp.asarray(ref_img[None, ...])
        
        # Use train=True and mutable to recalculate filters so gradients flow to all parameters
        feat_a, _ = model.apply({"params": params_val, **state}, a_j, train=True, mutable=list(state.keys()))
        feat_b, _ = model.apply({"params": params_val, **state}, b_j, train=True, mutable=list(state.keys()))
        
        # Differentiable RMSE over height, width, and channel dimensions
        diffs = jnp.sqrt(jnp.mean((feat_a - feat_b) ** 2, axis=(-3, -2, -1)) + 1e-8)
        
        # Pearson correlation
        corr = jax_pearson_correlation(diffs, a_interp_j)
        loss = -corr
        return loss, corr

    # Compute value and grads
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    print("Initial evaluation...")
    loss, init_corr = loss_fn(params)
    print(f"Initial Pearson correlation: {init_corr:.4f}")

    print("\nRunning optimization loop (10 steps)...")
    for i in range(10):
        (loss_val, corr_val), grads = grad_fn(params)
        
        # Update params
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        print(f"Step {i+1:02d} | Loss: {loss_val:.4f} | Correlation: {corr_val:.4f}")

    print("\nOptimization finished!")
    final_loss, final_corr = loss_fn(params)
    print(f"Final Pearson correlation: {final_corr:.4f}")
    print(f"Total improvement: {final_corr - init_corr:.4f}")

if __name__ == "__main__":
    main()
