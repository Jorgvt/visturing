# ResNet-18 Visturing Suite Evaluation Summary

This directory contains the psychophysical evaluation results of a pre-trained ResNet-18 model across multiple layers.

## Overview of Summary Tables

Below are the comparative tables for both the RMSE fit correlation ($\rho_p$) and Curve Order correlation ($\rho_k$) across all evaluated layers.

### 1. RMSE Fit Correlation ($\rho_p$) Summary
This table shows the Pearson correlation between model distances and human psychophysical response functions.

| Property        | layer1   |   layer2 |   layer3 |   layer4 |   avgpool |    fc |
|:----------------|:---------|---------:|---------:|---------:|----------:|------:|
| Prop. 1         | 0.60     |     0.46 |     0.41 |     0.35 |      0.33 |  0.25 |
| Prop. 2 achrom. | 0.93     |     0.81 |     0.78 |     0.77 |      0.78 |  0.69 |
| Prop. 2 chrom.  | 0.99     |     0.97 |     0.97 |     0.95 |      0.96 |  0.95 |
| Prop. 3 & 4     | -0.05    |     0.43 |     0.39 |     0.16 |     -0    | -0.2  |
| Prop. 5         | -        |    -0.31 |     0.67 |     0.75 |      0.79 |  0.75 |
| Prop. 6 & 7     | 0.09     |     0.45 |     0.43 |     0.65 |      0.66 |  0.65 |
| Prop. 8         | 0.47     |     0.78 |     0.62 |     0.85 |      0.87 |  0.83 |
| Prop. 9         | 0.47     |     0.78 |     0.62 |     0.85 |      0.87 |  0.83 |
| Prop. 10        | 0.47     |     0.78 |     0.62 |     0.85 |      0.87 |  0.83 |

### 2. Curve Order Correlation ($\rho_k$) Summary
This table shows the Spearman/Kendall-tau rank correlation for the ordering of the psychophysical functions.

| Property        | layer1                           | layer2                           | layer3                           | layer4                         | avgpool                        | fc                             |
|:----------------|:---------------------------------|:---------------------------------|:---------------------------------|:-------------------------------|:-------------------------------|:-------------------------------|
| Prop. 1         | -                                | -                                | -                                | -                              | -                              | -                              |
| Prop. 2 achrom. | 0.53                             | 0.58                             | 0.49                             | 0.61                           | 0.34                           | 0.13                           |
| Prop. 2 chrom.  | RG: 1.00 | YB: 1.00              | RG: 1.00 | YB: 1.00              | RG: 0.81 | YB: 0.97              | RG: 0.62 | YB: 0.81            | RG: 0.54 | YB: 0.83            | RG: 0.49 | YB: 0.62            |
| Prop. 3 & 4     | 0.27                             | 0.68                             | 0.94                             | 0.67                           | 0.54                           | 0.42                           |
| Prop. 5         | -0.62                            | -0.62                            | 0.52                             | 0.92                           | 0.94                           | 0.89                           |
| Prop. 6 & 7     | A: -0.23 | RG: -0.74 | YB: -0.77 | A: -0.05 | RG: -0.58 | YB: -0.59 | A: -0.23 | RG: -0.66 | YB: -0.58 | A: -0.36 | RG: 0.11 | YB: 0.06 | A: -0.28 | RG: 0.04 | YB: 0.00 | A: -0.23 | RG: 0.10 | YB: 0.11 |
| Prop. 8         | Low f: 0.89 | High f: 1.00       | Low f: 0.93 | High f: 1.00       | Low f: 0.64 | High f: 1.00       | Low f: 0.61 | High f: 1.00     | Low f: 0.56 | High f: 0.93     | Low f: 0.58 | High f: 0.92     |
| Prop. 9         | Low f: -0.24 | High f: -0.33     | Low f: -0.11 | High f: -0.73     | Low f: -0.20 | High f: -0.33     | Low f: -0.40 | High f: -0.67   | Low f: -0.47 | High f: -0.34   | Low f: -0.30 | High f: -0.33   |
| Prop. 10        | Low f: 0.22 | High f: 0.83       | Low f: 0.63 | High f: 0.76       | Low f: 0.80 | High f: 0.73       | Low f: 0.38 | High f: 0.81     | Low f: 0.33 | High f: 0.88     | Low f: 0.30 | High f: 0.90     |

## Layer-wise Detailed Tables

### ResNet-18 Layer: `layer1`

| Property        | RMSE fit (ρ_p)   | Curve Order (ρ_k)                |
|:----------------|:-----------------|:---------------------------------|
| Prop. 1         | 0.60             | -                                |
| Prop. 2 achrom. | 0.93             | 0.53                             |
| Prop. 2 chrom.  | 0.99             | RG: 1.00 | YB: 1.00              |
| Prop. 3 & 4     | -0.05            | 0.27                             |
| Prop. 5         | -                | -0.62                            |
| Prop. 6 & 7     | 0.09             | A: -0.23 | RG: -0.74 | YB: -0.77 |
| Prop. 8         | 0.47             | Low f: 0.89 | High f: 1.00       |
| Prop. 9         | 0.47             | Low f: -0.24 | High f: -0.33     |
| Prop. 10        | 0.47             | Low f: 0.22 | High f: 0.83       |

### ResNet-18 Layer: `layer2`

| Property        |   RMSE fit (ρ_p) | Curve Order (ρ_k)                |
|:----------------|-----------------:|:---------------------------------|
| Prop. 1         |             0.46 | -                                |
| Prop. 2 achrom. |             0.81 | 0.58                             |
| Prop. 2 chrom.  |             0.97 | RG: 1.00 | YB: 1.00              |
| Prop. 3 & 4     |             0.43 | 0.68                             |
| Prop. 5         |            -0.31 | -0.62                            |
| Prop. 6 & 7     |             0.45 | A: -0.05 | RG: -0.58 | YB: -0.59 |
| Prop. 8         |             0.78 | Low f: 0.93 | High f: 1.00       |
| Prop. 9         |             0.78 | Low f: -0.11 | High f: -0.73     |
| Prop. 10        |             0.78 | Low f: 0.63 | High f: 0.76       |

### ResNet-18 Layer: `layer3`

| Property        |   RMSE fit (ρ_p) | Curve Order (ρ_k)                |
|:----------------|-----------------:|:---------------------------------|
| Prop. 1         |             0.41 | -                                |
| Prop. 2 achrom. |             0.78 | 0.49                             |
| Prop. 2 chrom.  |             0.97 | RG: 0.81 | YB: 0.97              |
| Prop. 3 & 4     |             0.39 | 0.94                             |
| Prop. 5         |             0.67 | 0.52                             |
| Prop. 6 & 7     |             0.43 | A: -0.23 | RG: -0.66 | YB: -0.58 |
| Prop. 8         |             0.62 | Low f: 0.64 | High f: 1.00       |
| Prop. 9         |             0.62 | Low f: -0.20 | High f: -0.33     |
| Prop. 10        |             0.62 | Low f: 0.80 | High f: 0.73       |

### ResNet-18 Layer: `layer4`

| Property        |   RMSE fit (ρ_p) | Curve Order (ρ_k)              |
|:----------------|-----------------:|:-------------------------------|
| Prop. 1         |             0.35 | -                              |
| Prop. 2 achrom. |             0.77 | 0.61                           |
| Prop. 2 chrom.  |             0.95 | RG: 0.62 | YB: 0.81            |
| Prop. 3 & 4     |             0.16 | 0.67                           |
| Prop. 5         |             0.75 | 0.92                           |
| Prop. 6 & 7     |             0.65 | A: -0.36 | RG: 0.11 | YB: 0.06 |
| Prop. 8         |             0.85 | Low f: 0.61 | High f: 1.00     |
| Prop. 9         |             0.85 | Low f: -0.40 | High f: -0.67   |
| Prop. 10        |             0.85 | Low f: 0.38 | High f: 0.81     |

### ResNet-18 Layer: `avgpool`

| Property        |   RMSE fit (ρ_p) | Curve Order (ρ_k)              |
|:----------------|-----------------:|:-------------------------------|
| Prop. 1         |             0.33 | -                              |
| Prop. 2 achrom. |             0.78 | 0.34                           |
| Prop. 2 chrom.  |             0.96 | RG: 0.54 | YB: 0.83            |
| Prop. 3 & 4     |            -0    | 0.54                           |
| Prop. 5         |             0.79 | 0.94                           |
| Prop. 6 & 7     |             0.66 | A: -0.28 | RG: 0.04 | YB: 0.00 |
| Prop. 8         |             0.87 | Low f: 0.56 | High f: 0.93     |
| Prop. 9         |             0.87 | Low f: -0.47 | High f: -0.34   |
| Prop. 10        |             0.87 | Low f: 0.33 | High f: 0.88     |

### ResNet-18 Layer: `fc`

| Property        |   RMSE fit (ρ_p) | Curve Order (ρ_k)              |
|:----------------|-----------------:|:-------------------------------|
| Prop. 1         |             0.25 | -                              |
| Prop. 2 achrom. |             0.69 | 0.13                           |
| Prop. 2 chrom.  |             0.95 | RG: 0.49 | YB: 0.62            |
| Prop. 3 & 4     |            -0.2  | 0.42                           |
| Prop. 5         |             0.75 | 0.89                           |
| Prop. 6 & 7     |             0.65 | A: -0.23 | RG: 0.10 | YB: 0.11 |
| Prop. 8         |             0.83 | Low f: 0.58 | High f: 0.92     |
| Prop. 9         |             0.83 | Low f: -0.30 | High f: -0.33   |
| Prop. 10        |             0.83 | Low f: 0.30 | High f: 0.90     |

