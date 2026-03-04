import wget
from zipfile import ZipFile
import os

import numpy as np
import pandas as pd

from . import prop1
from . import prop2
from . import prop3_4
from . import prop5
from . import prop6_7
from . import prop8
from . import prop9
from . import prop10

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
    """Takes as input the output of the `evaluate_all` function and returns a table like in the paper.""""

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
