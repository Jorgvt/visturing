# ResNet-18 TID2008 & TID2013 Evaluation Summary

This document summarizes the performance of a pretrained ResNet-18 model evaluated at multiple layers on the TID2008 and TID2013 image quality assessment benchmarks.

## TID2008 Evaluation Results

The table below reports Pearson (PLCC), Spearman (SROCC), and Kendall (KROCC) correlation coefficients between feature-based model distances and subjective Mean Opinion Scores (MOS).

| Layer   |      PLCC |     SROCC |     KROCC |   |PLCC| |   |SROCC| |   |KROCC| |
|:--------|----------:|----------:|----------:|---------:|----------:|----------:|
| layer1  | -0.721971 | -0.725225 | -0.539558 | 0.721971 |  0.725225 |  0.539558 |
| layer2  | -0.767076 | -0.753995 | -0.571058 | 0.767076 |  0.753995 |  0.571058 |
| layer3  | -0.785595 | -0.762739 | -0.576662 | 0.785595 |  0.762739 |  0.576662 |
| layer4  | -0.78194  | -0.758429 | -0.567586 | 0.78194  |  0.758429 |  0.567586 |
| avgpool | -0.767177 | -0.717575 | -0.531173 | 0.767177 |  0.717575 |  0.531173 |
| fc      | -0.756829 | -0.704501 | -0.519776 | 0.756829 |  0.704501 |  0.519776 |

## TID2013 Evaluation Results

The table below reports the same metrics on the larger TID2013 benchmark dataset.

| Layer   |      PLCC |     SROCC |     KROCC |   |PLCC| |   |SROCC| |   |KROCC| |
|:--------|----------:|----------:|----------:|---------:|----------:|----------:|
| layer1  | -0.749763 | -0.731866 | -0.552192 | 0.749763 |  0.731866 |  0.552192 |
| layer2  | -0.773178 | -0.743486 | -0.565104 | 0.773178 |  0.743486 |  0.565104 |
| layer3  | -0.791415 | -0.758427 | -0.572291 | 0.791415 |  0.758427 |  0.572291 |
| layer4  | -0.787857 | -0.763407 | -0.570349 | 0.787857 |  0.763407 |  0.570349 |
| avgpool | -0.780783 | -0.738697 | -0.547799 | 0.780783 |  0.738697 |  0.547799 |
| fc      | -0.769159 | -0.724882 | -0.535514 | 0.769159 |  0.724882 |  0.535514 |

> [!NOTE]
> Since model distance is inversely related to perceived image quality (MOS), the raw correlation values (PLCC, SROCC, KROCC) are negative. The absolute values ($|\text{PLCC}|$, $|\text{SROCC}|$, $|\text{KROCC}|$) represent the strength of the alignment.

