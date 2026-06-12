
# Visturing: A Turing Test for Artificial Nets devoted to Vision
This repository contains the source code, datasets, and experiments associated with the research paper "A Turing Test for Artificial Nets devoted to Vision", published in Frontiers in Artificial Intelligence (2025).

The project implements a comprehensive evaluation framework (a visual "Turing Test") to assess whether Artificial Neural Networks (ANNs) exhibit the same low-level spatio-chromatic properties as the human Retina-V1 pathway.

## Overview

Deep networks have achieved remarkable success in computer vision, but do they actually "see" like humans? This work proposes that for an ANN to be considered a valid biological model, it must pass rigorous psychophysical and physiological tests, not just achieve high accuracy on segmentation or classification tasks.

Key Contributions:

The Test Suite: A collection of 10 qualitative and quantitative tests covering fundamental properties of the visual system (e.g., contrast sensitivity, masking, frequency tuning).

Model Comparison: We evaluate three distinct modeling approaches:

Parametric Model: Optimized via Maximum Differentiation (MaxDiff).

Non-Parametric Model (PerceptNet): Optimized to maximize correlation with human subjective distortion.

Segmentation Model: The same architecture as PerceptNet, but trained for a technical segmentation task.

Results: The code reproduces the paper's findings, showing that models trained on human perception (PerceptNet) align significantly better with biological behavior than those trained for pure computer vision tasks.

## Installation

To reproduce the experiments, clone this repository and install the required dependencies. It is recommended to use a virtual environment (Anaconda or venv).

```Bash
# Install the repository
pip install git+https://github.com/Jorgvt/visturing.git

```


## Usage & Reproduction
The primary results and figures from the paper are generated using Jupyter Notebooks.

To evaluate some models check the folder *use_examples*

### Custom Evaluation Configurations (e.g., Custom Image Sizes)

You can customize the parameters used during generative evaluation (such as changing the evaluation image size from `(128, 128)` to `(256, 256)`) by passing a configuration override dictionary to the `configs` argument of `evaluate_all_gen`:

```python
from visturing.properties.utils import evaluate_all_gen, build_evaluation_table_gen

# Define custom configuration overrides for generative properties
# (Note: prop1 loads static dataset files from disk, so it cannot be resized dynamically)
custom_configs = {
    "prop2": {
        "img_size": (256, 256),
        "square_size": (128, 128)  # Scaled center patch scale proportionally
    },
    "prop3_4": {"img_size": (256, 256)},
    "prop5":   {"img_size": (256, 256)},
    "prop6_7": {"img_size": (256, 256)},
    "prop8":   {"img_size": (256, 256)},
    "prop9":   {"img_size": (256, 256)},
    "prop10":  {"img_size": (256, 256)},
}

# Run the evaluation
results = evaluate_all_gen(
    calculate_diffs=calculate_diffs,
    configs=custom_configs,
    batch_size=16,
    verbose=True
)

# Print the formatted markdown evaluation table
print(build_evaluation_table_gen(results))
```

## Repository Structure

use_examples/: Examples to evaluate classical models using the library. It automatically downloads the data in /Data if necessary.

visturing/: Scripts implementing the 10 psychophysical/physiological tests.


## Citation
If you use this code, data, or methodology in your research, please cite the original article:

```
@article{vila2025turing,
  title={A Turing Test for Artificial Nets devoted to Vision},
  author={Vila-Tomás, Jorge and Hernández-Cámara, Pablo and Li, Qiang and Laparra, Valero and Malo, Jesús},
  journal={Frontiers in Artificial Intelligence},
  year={2025},
  doi={10.3389/frai.2025.1665874},
  url={https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1665874/abstract}
}
```


👥 Authors
Jorge Vila-Tomás (Universitat de València)

Pablo Hernández-Cámara (Universitat de València)

Qiang Li (Georgia Institute of Technology)

Valero Laparra (Universitat de València)

Jesús Malo (Universitat de València) - Corresponding Author

For any questions regarding the code or the paper, please open an Issue in this repository.

Image Processing Lab (IPL), Universitat de València.