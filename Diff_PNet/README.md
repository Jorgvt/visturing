# JAX PerceptNet Optimization Experiments: Properties 1 to 10

This directory replicates our optimization setup under the **PerceptNet** model. It demonstrates how to load pretrained parameters and optimize PerceptNet's parameters to maximize its alignment (correlation) with human visibility data.

The setup is organized into two subfolders:
- **[`weighted/`](file:///Users/jorgvt/Developer/visturing/Diff_PNet/weighted)**: Optimizes parameters to maximize the global **weighted** Pearson correlation.
- **[`non_weighted/`](file:///Users/jorgvt/Developer/visturing/Diff_PNet/non_weighted)**: Optimizes parameters to maximize the global **non-weighted** Pearson correlation.

---

## 📂 Directory Structure & Scripts

Both folders contain optimization scripts for all 8 properties:
- **`optimize_prop1.py`**: Spectral sensitivities (non-weighted ground truth only).
- **`optimize_prop2.py`**: Color discrimination.
- **`optimize_prop3_4.py`**: Contrast Sensitivity Function (CSF).
- **`optimize_prop5.py`**: Masked CSF (Campbell & Blakemore).
- **`optimize_prop6_7.py`**: Color-sensitive CSF / masking.
- **`optimize_prop8.py`**: Spatio-temporal masking CSF.
- **`optimize_prop9.py`**: Masking spatial frequency-dependent.
- **`optimize_prop10.py`**: Masking orientation-dependent.

---

## 🚀 How to Run

Ensure JAX, Flax, Optax, and `paramperceptnet` are installed. Run any script with:

```bash
# Example: Run Property 5 weighted optimization
uv run --group jax python Diff_PNet/weighted/optimize_prop5.py

# Example: Run Property 10 non-weighted optimization
uv run --group jax python Diff_PNet/non_weighted/optimize_prop10.py
```

### ⚙️ Command-Line Arguments

All scripts accept CLI arguments to configure the execution:
- `--batch_size`: The batch size to use during evaluation (default: `None`, which evaluates in a single batch).
- `--iterations`: The number of training/optimization iterations (default: `10`).

Example with CLI arguments:
```bash
uv run --group jax python Diff_PNet/weighted/optimize_prop3_4.py --batch_size 32 --iterations 25
```

At the end of training, the script will automatically save the trained parameters and filter states inside the same directory under a file named:
`model_pnet_prop{N}.pkl`

---

## 🔑 Crucial Implementation Details

### 1. State/Filter Initialization
PerceptNet relies on precalculated filters stored in Flax's `state` collection (specifically, inside `precalc_filter`). To populate these collections before optimization, the scripts execute a dummy forward pass once at the beginning:
```python
model, variables = load_param_pretrained()
params = variables["params"]
state = variables["state"]

# Run dummy forward pass to build filter variables
dummy_x = jnp.zeros((1, 128, 128, 3))
_, state = model.apply({"params": params, **state}, dummy_x, train=True, mutable=list(state.keys()))
```

### 2. Differentiable Filter Recalculation
To allow gradients to flow back to parameters that define filter shapes (like `CenterSurroundLogSigmaK_0`), the model is evaluated with `train=True` and `mutable=list(state.keys())` inside the loss function:
```python
feat_a, _ = model.apply({"params": params_val, **state}, a_j, train=True, mutable=list(state.keys()))
```
Using `train=True` triggers dynamic recalculation of filters from the current model parameters `params_val`, keeping JAX's gradient tracking fully connected.
