import wget
from zipfile import ZipFile
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

@dataclass
class EvaluationResult:
    results: Dict[str, Any]
    correlations: Dict[str, Any]
    stimuli: Optional[Dict[str, Any]] = None
    gt: Optional[Dict[str, Any]] = None
    freqs: Optional[Any] = None

def run_batched(calculate_diffs, a, b, batch_size: Optional[int] = None, show_progress: bool = False, desc: str = "Evaluating batches"):
    """Slices the inputs into batches of `batch_size` and runs `calculate_diffs` on each batch.
    
    Supports NumPy, JAX, and PyTorch outputs.
    """
    if batch_size is None or batch_size <= 0 or len(a) <= batch_size:
        if show_progress:
            from tqdm import tqdm
            with tqdm(total=1, desc=desc, unit="batch") as pbar:
                res = calculate_diffs(a, b)
                pbar.update(1)
            return res
        return calculate_diffs(a, b)
    
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(a), desc=desc, unit="stimuli")
    else:
        pbar = None

    diffs = []
    for i in range(0, len(a), batch_size):
        chunk_a = a[i : i + batch_size]
        chunk_b = b[i : i + batch_size]
        diffs.append(calculate_diffs(chunk_a, chunk_b))
        if pbar:
            pbar.update(len(chunk_a))
            
    if pbar:
        pbar.close()
    
    first_diff = diffs[0]
    if hasattr(first_diff, "requires_grad") or (hasattr(first_diff, "__class__") and first_diff.__class__.__name__ == 'Tensor'):
        import torch
        # Ensure they are tensors and stack/concatenate
        return torch.cat([torch.as_tensor(d) for d in diffs], dim=0)
    elif hasattr(first_diff, "__class__") and 'jax' in first_diff.__class__.__module__:
        import jax.numpy as jnp
        return jnp.concatenate(diffs, axis=0)
    else:
        return np.concatenate([np.asarray(d) for d in diffs])


class ArrayApi:
    def __init__(self, xp):
        self.xp = xp
        self.is_torch = (xp.__name__ == 'torch') if hasattr(xp, '__name__') else False
        
    def mean(self, x, axis=None):
        return self.xp.mean(x, dim=axis) if self.is_torch else self.xp.mean(x, axis=axis)
        
    def sum(self, x, axis=None):
        return self.xp.sum(x, dim=axis) if self.is_torch else self.xp.sum(x, axis=axis)
        
    def stack(self, arrays, axis=0):
        return self.xp.stack(arrays, dim=axis) if self.is_torch else self.xp.stack(arrays, axis=axis)
        
    def repeat(self, x, repeats, axis=None):
        return self.xp.repeat_interleave(x, repeats, dim=axis) if self.is_torch else self.xp.repeat(x, repeats, axis=axis)
        
    def transpose(self, x, axes=None):
        if self.is_torch:
            return x.permute(*axes) if axes is not None else x.t()
        return x.transpose(axes) if axes is not None else x.T

    def ravel(self, x):
        return x.flatten() if self.is_torch else x.ravel()

    def broadcast_to(self, x, shape):
        return self.xp.broadcast_to(x, shape)

    def ones_like(self, x):
        return self.xp.ones_like(x)

    def sqrt(self, x):
        return self.xp.sqrt(x)

    def concatenate(self, arrays, axis=0):
        return self.xp.cat(arrays, dim=axis) if self.is_torch else self.xp.concatenate(arrays, axis=axis)

    def asarray(self, x):
        return self.xp.as_tensor(x) if self.is_torch else self.xp.asarray(x)


def weighted_pearson_correlation(x, y, w, xp=np):
    """
    Calculate the weighted Pearson correlation coefficient.
    Supports NumPy, JAX, and PyTorch.
    """
    if not isinstance(xp, ArrayApi):
        xp = ArrayApi(xp)
        
    w = w / xp.sum(w)
    mean_x = xp.sum(w * x)
    mean_y = xp.sum(w * y)
    dev_x = x - mean_x
    dev_y = y - mean_y
    cov_xy = xp.sum(w * dev_x * dev_y)
    var_x = xp.sum(w * dev_x ** 2)
    var_y = xp.sum(w * dev_y ** 2)
    return cov_xy / xp.sqrt(var_x * var_y)


def download_ground_truth(data_path, # Path to download the data
                  ):
    data_url = "https://zenodo.org/records/17700252/files/ground_truth.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(data_path, "ground_truth")

def evaluate_all(calculate_diffs,
                 data_path, # Path to the root directory
                 gt_path, # Path to the ground truth
                 ):
    from . import prop1
    from . import prop2
    from . import prop3_4
    from . import prop5
    from . import prop6_7
    from . import prop8
    from . import prop9
    from . import prop10

    if not os.path.exists(os.path.join(gt_path, "ground_truth")):
        gt_path = download_ground_truth(gt_path)
    else:
        gt_path = os.path.join(gt_path, "ground_truth")

    results = {}
    results["prop1"] = prop1.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_1"), gt_path)
    print('prop1 done')
    results["prop2"] = prop2.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_2"), gt_path)
    print('prop2 done')
    results["prop3_4"] = prop3_4.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_3_4"), gt_path)
    print('prop3_4 done')
    results["prop5"] = prop5.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_5"), gt_path)
    print('prop5 done')
    results["prop6_7"] = prop6_7.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_6_7"), gt_path)
    print('prop6_7 done')
    results["prop8"] = prop8.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_8"), gt_path)
    print('prop8 done')
    results["prop9"] = prop9.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_9"), gt_path)
    print('prop9 done')
    results["prop10"] = prop10.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_10"), gt_path)
    print('prop10 done')

    return results


def build_evaluation_table(data):
    """Takes as input the output of the `evaluate_all` function and returns a table like in the paper."""

    # Extract only the correlations to ease processing
    data = {p:c["correlations"] for p, c in data.items()}


    # Helper to safely extract and format values
    def get_val(val):
        return f"{val:.2f}" if val is not None and not np.isnan(val) else "-"

    rows = []

    # Prop. 1
    rows.append({
        "Property": "Prop. 1",
        "RMSE fit (ρ_p)": get_val(data.get('prop1', {}).get('pearson')),
        "Curve Order (ρ_k)": "-" # Not present in the provided JSON for Prop 1
    })

    # Prop. 2 achrom.
    rows.append({
        "Property": "Prop. 2 achrom.",
        "RMSE fit (ρ_p)": get_val(data.get('prop2', {}).get('pearson_achrom')),
        "Curve Order (ρ_k)": get_val(data.get('prop2', {}).get('kendall', {}).get('achrom', {}).get('kendall'))
    })

    # Prop. 2 chrom.
    p2_chrom_rg = get_val(data.get('prop2', {}).get('kendall', {}).get('red_green', {}).get('kendall'))
    p2_chrom_yb = get_val(data.get('prop2', {}).get('kendall', {}).get('yellow_blue', {}).get('kendall'))
    rows.append({
        "Property": "Prop. 2 chrom.",
        "RMSE fit (ρ_p)": get_val(data.get('prop2', {}).get('pearson_chrom')),
        "Curve Order (ρ_k)": f"RG: {p2_chrom_rg} | YB: {p2_chrom_yb}"
    })

    # Prop. 3 & 4
    rows.append({
        "Property": "Prop. 3 & 4",
        "RMSE fit (ρ_p)": get_val(data.get('prop3_4', {}).get('pearson')),
        "Curve Order (ρ_k)": get_val(data.get('prop3_4', {}).get('kendall', {}).get('kendall'))
    })

    # Prop. 5
    rows.append({
        "Property": "Prop. 5",
        "RMSE fit (ρ_p)": get_val(data.get('prop5', {}).get('pearson')),
        "Curve Order (ρ_k)": get_val(data.get('prop5', {}).get('kendall', {}).get('kendall'))
    })

    # Prop. 6 & 7
    p67_a = get_val(data.get('prop6_7', {}).get('kendall', {}).get('achrom', {}).get('kendall'))
    p67_rg = get_val(data.get('prop6_7', {}).get('kendall', {}).get('red_green', {}).get('kendall'))
    p67_yb = get_val(data.get('prop6_7', {}).get('kendall', {}).get('yellow_blue', {}).get('kendall'))
    rows.append({
        "Property": "Prop. 6 & 7",
        "RMSE fit (ρ_p)": get_val(data.get('prop6_7', {}).get('pearson')),
        "Curve Order (ρ_k)": f"A: {p67_a} | RG: {p67_rg} | YB: {p67_yb}"
    })

    # Props 8, 9, 10 (Note: In the image, 8, 9, and 10 share the same RMSE fit. 
    # Your data only has it under prop8, so we'll reuse it for 9 and 10).
    shared_rmse = get_val(data.get('prop8', {}).get('pearson'))

    for prop_key, prop_name in [('prop8', 'Prop. 8'), ('prop9', 'Prop. 9'), ('prop10', 'Prop. 10')]:
        low_f = get_val(data.get(prop_key, {}).get('kendall', {}).get('low', {}).get('kendall'))
        high_f = get_val(data.get(prop_key, {}).get('kendall', {}).get('high', {}).get('kendall'))
        
        rows.append({
            "Property": prop_name,
            "RMSE fit (ρ_p)": shared_rmse,
            "Curve Order (ρ_k)": f"Low f: {low_f} | High f: {high_f}"
        })

    # Convert to DataFrame for a clean tabular format
    df = pd.DataFrame(rows)

    return df

def extract_numbers_from_table(table_results):
    numbers = []
    for e in table_results.to_numpy().ravel():
        a = re.findall("[-+]?\d+\.\d+", e)
        if len(a) > 0:
            for n in a:
                numbers.append(float(n))
    return numbers


def evaluate_all_gen(calculate_diffs,
                     batch_size: int | None = None,
                     configs: dict | None = None,
                     verbose: bool = False,
                     show_property_progress: bool = False,
                     data_path: str = "./", # path to the root directory
                     gt_path: str = "./", # path to the ground truth
                     ):
    """Evaluates all available generative properties using their default configurations.
    
    Args:
        calculate_diffs: User-provided function to calculate differences.
        batch_size: Optional batch size for deep learning model evaluations.
        configs: Optional dictionary to override default configurations for properties.
        verbose: If True, displays a global progress bar over properties.
        show_property_progress: If True, displays progress bars within each property evaluation.
    """
    from .config import DEFAULT_CONFIGS
    from . import prop1
    from . import prop2
    from . import prop3_4
    from . import prop5
    from . import prop6_7
    from . import prop8
    from . import prop9
    from . import prop10

    if not os.path.exists(os.path.join(gt_path, "ground_truth")):
        gt_path = download_ground_truth(gt_path)
    else:
        gt_path = os.path.join(gt_path, "ground_truth")

    active_configs = {}
    for prop_name, default_cfg in DEFAULT_CONFIGS.items():
        active_configs[prop_name] = default_cfg.copy()
        if configs and prop_name in configs:
            active_configs[prop_name].update(configs[prop_name])
            
    properties_to_evaluate = [
        ("prop1", prop1),
        ("prop2", prop2),
        ("prop3_4", prop3_4),
        ("prop5", prop5),
        ("prop6_7", prop6_7),
        ("prop8", prop8),
        ("prop9", prop9),
        ("prop10", prop10),
    ]
    
    if verbose:
        from tqdm import tqdm
        pbar = tqdm(properties_to_evaluate, desc="Evaluating all properties")
    else:
        pbar = properties_to_evaluate
        
    results = {}
    for name, prop_mod in pbar:
        if verbose:
            pbar.set_postfix_str(f"Running {name}")
            
        if name == "prop1":
            res_prop1 = prop_mod.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_1"), gt_path)
            corr = res_prop1["correlations"]["pearson"]
            results[name] = EvaluationResult(
                results=res_prop1,
                correlations={"global": corr, "achrom": corr},
                gt=None,
                freqs=res_prop1.get("lambdas")
            )
            continue

        results[name] = prop_mod.evaluate_gen(
            calculate_diffs,
            return_gt=True,
            batch_size=batch_size,
            verbose=show_property_progress,
            **active_configs[name]
        )
    
    return results


def build_evaluation_table_gen(data):
    """Takes as input the output of the `evaluate_all_gen` function and returns a table of Pearson correlations."""
    def get_corr_val(val):
        if val is None:
            return "-"
        if isinstance(val, (tuple, list, np.ndarray)):
            val = val[0]
        return f"{val:.2f}" if not np.isnan(val) else "-"

    rows = []

    for prop_key, prop_name in [
        ("prop1", "Prop. 1"),
        ("prop2", "Prop. 2"),
        ("prop3_4", "Prop. 3 & 4"),
        ("prop5", "Prop. 5"),
        ("prop6_7", "Prop. 6 & 7"),
        ("prop8", "Prop. 8"),
        ("prop9", "Prop. 9"),
        ("prop10", "Prop. 10"),
    ]:
        prop_data = data.get(prop_key)
        if prop_data is not None and hasattr(prop_data, "correlations"):
            corrs = prop_data.correlations
            if "non-weighted" in corrs or "weighted" in corrs:
                for w_key, w_name in [("non-weighted", "Non-weighted"), ("weighted", "Weighted")]:
                    sub_corrs = corrs.get(w_key, {})
                    rows.append({
                        "Property": f"{prop_name} ({w_name})",
                        "Global (r)": get_corr_val(sub_corrs.get("global")),
                        "Achromatic (r)": get_corr_val(sub_corrs.get("achrom")),
                        "Red-Green (r)": get_corr_val(sub_corrs.get("red-green")),
                        "Yellow-Blue (r)": get_corr_val(sub_corrs.get("yellow-blue")),
                    })
            else:
                rows.append({
                    "Property": prop_name,
                    "Global (r)": get_corr_val(corrs.get("global")),
                    "Achromatic (r)": get_corr_val(corrs.get("achrom")),
                    "Red-Green (r)": get_corr_val(corrs.get("red-green")),
                    "Yellow-Blue (r)": get_corr_val(corrs.get("yellow-blue")),
                })
        else:
            rows.append({
                "Property": prop_name,
                "Global (r)": "-",
                "Achromatic (r)": "-",
                "Red-Green (r)": "-",
                "Yellow-Blue (r)": "-",
            })

    return pd.DataFrame(rows)
