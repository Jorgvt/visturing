import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from paramperceptnet.pretrained import load_param_pretrained
from visturing.properties import prop3_4
from visturing.properties.config import default_prop3_4_config

def main():
    import argparse
    parser = argparse.ArgumentParser(description="JAX Optimization Experiment")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    args = parser.parse_args()

    print("Starting JAX PerceptNet optimization experiment on Prop. 3 & 4 (CSF) using WEIGHTED correlation...")

    # Load pretrained PerceptNet model
    model, variables = load_param_pretrained()
    params = variables["params"]
    state = variables["state"]

    # Run a dummy forward pass to populate precalc_filter variables in state
    dummy_x = jnp.zeros((1, 128, 128, 3))
    _, state = model.apply({"params": params, **state}, dummy_x, train=True, mutable=list(state.keys()))
    print("Precalculated filters populated in state!")

    # Initialize Optax optimizer (Adam)
    tx = optax.adam(learning_rate=1e-4)
    opt_state = tx.init(params)

    # Define loss function
    def loss_fn(params_val):
        def calculate_diffs(a, b):
            a_j = jnp.asarray(a)
            b_j = jnp.asarray(b)
            # Use train=True and mutable to recalculate filters so gradients flow to all parameters
            feat_a, _ = model.apply({"params": params_val, **state}, a_j, train=True, mutable=list(state.keys()))
            feat_b, _ = model.apply({"params": params_val, **state}, b_j, train=True, mutable=list(state.keys()))
            # Differentiable RMSE over height, width, and channel dimensions
            diffs = jnp.sqrt(jnp.mean((feat_a - feat_b) ** 2, axis=(-3, -2, -1)) + 1e-8)
            return diffs

        # Call evaluate_gen with default configuration
        res = prop3_4.evaluate_gen(
            calculate_diffs,
            xp=jnp,
            batch_size=args.batch_size,
            verbose=False,
            **default_prop3_4_config
        )
        
        # Optimize weighted global correlation
        corr = res.correlations["weighted"]["global"]
        loss = -corr
        return loss, corr

    # Compute value and grads
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    print("Initial evaluation...")
    loss, init_corr = loss_fn(params)
    print(f"Initial global weighted correlation: {init_corr:.4f}")

    print(f"\nRunning optimization loop ({args.iterations} steps)...")
    for i in range(args.iterations):
        (loss_val, corr_val), grads = grad_fn(params)
        
        # Update params
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        print(f"Step {i+1:02d} | Loss: {loss_val:.4f} | Correlation: {corr_val:.4f}")

    print("\nOptimization finished!")
    final_loss, final_corr = loss_fn(params)
    print(f"Final global weighted correlation: {final_corr:.4f}")
    print(f"Total improvement: {final_corr - init_corr:.4f}")

    print("\nSaving trained model variables...")
    import pickle
    save_path = os.path.join(os.path.dirname(__file__), "model_pnet_prop3_4.pkl")
    variables_to_save = {"params": params, "state": state}
    with open(save_path, "wb") as f_save:
        pickle.dump(variables_to_save, f_save)
    print(f"Saved variables to {save_path}")

if __name__ == "__main__":
    main()
