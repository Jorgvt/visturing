
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

Repository Structure

use_examples/: Examples to evaluate classical models using the library. It aomatically downloads the data in /Data if necessary.

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