import wget
from zipfile import ZipFile
import os

from ..properties import prop1
from ..properties import prop2
from ..properties import prop3_4
from ..properties import prop5
from ..properties import prop6_7
from ..properties import prop8
from ..properties import prop9
from ..properties import prop10

def download_ground_truth(data_path, # Path to download the data
                  ):
    data_url = "https://zenodo.org/records/17700252/files/ground_truth.zip"
    path = wget.download(data_url)
    with ZipFile(path) as zipObj:
        zipObj.extractall(data_path)
    os.remove(path)
    return os.path.join(gt_path, "ground_truth")

def evaluate_all(calculate_diffs,
                 data_path, # Path to the root directory
                 gt_path,
                 ):

    if not os.path.exists(os.path.join(gt_path, "ground_truth")):
        gt_path = download_ground_truth(gt_path)

    results = {}
    results["prop1"] = prop1.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_1"), gt_path)
    results["prop2"] = prop2.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_2"), gt_path)
    results["prop3_4"] = prop3_4.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_3_4"), gt_path)
    results["prop5"] = prop5.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_5"), gt_path)
    results["prop6_7"] = prop6_7.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_6_7"), gt_path)
    results["prop8"] = prop8.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_8"), gt_path)
    results["prop9"] = prop9.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_9"), gt_path)
    results["prop10"] = prop10.evaluate(calculate_diffs, os.path.join(data_path, "Experiment_10"), gt_path)

    return results
