import wget
from zipfile import ZipFile
import os

import wget
from zipfile import ZipFile
import os

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
