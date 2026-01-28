import numpy as np
import scipy.stats as stats

def calculate_spearman(experimental_curve, ideal_ordering):
    ordered_curves = (-experimental_curve).argsort(axis=0)
    ideal_ordering = np.array(ideal_ordering)[:,None].repeat(ordered_curves.shape[1], axis=1)
    return calculate_correlations(ordered_curves, ideal_ordering)

def calculate_correlations(ground_truth, experimental):
    return {
        "spearman": stats.spearmanr(ground_truth.ravel(),
                                    experimental.ravel())[0],
        "kendall": stats.kendalltau(ground_truth.ravel(),
                                    experimental.ravel())[0],
        "pearson": stats.pearsonr(ground_truth.ravel(),
                                    experimental.ravel())[0],
    }

def calculate_correlations_with_ground_truth(experimental_curve, ground_truth):
    gt_ordering = (-ground_truth).argsort(axis=0)
    e_ordering = (-experimental_curve).argsort(axis=0)
    return calculate_correlations(gt_ordering.ravel(), e_ordering.ravel())

def compare_ranges(x1, x2):
    m, M = False, False
    if x1.min() <= x2.min():
        m = True
    if x1.max() >= x2.max():
        M = True
    return all((m, M))

def prepare_data(x_e, y_e, x_gt, y_gt):
    """Prepares the data to be compared via correlations."""
    if gt_shorter:=compare_ranges(x_e, x_gt):
        x_wider = x_e
        y_wider = y_e
        x_shorter = x_gt
        y_shorter = y_gt
    else:
        x_wider = x_gt
        y_wider = y_gt
        x_shorter = x_e
        y_shorter = y_e

    # print(x_wider.shape, y_wider.shape, x_shorter.shape, y_shorter.shape)
    # if len(y_wider) > 1:
    #     y_wider_interp = np.array([np.interp(x_shorter, x_wider, row) for row in y_wider])
    # else:
    #     y_wider_interp = np.interp(x_shorter, x_wider, y_wider)
    y_wider_interp = np.interp(x_shorter, x_wider, y_wider)
    if gt_shorter:
        return x_shorter, y_wider_interp, x_shorter, y_shorter
    else:
        return x_shorter, y_shorter, x_shorter, y_wider_interp
    
def prepare_and_correlate(x_e, y_e, x_gt, y_gt):
    x_e, y_e, x_gt, y_gt = prepare_data(x_e, y_e, x_gt, y_gt)
    return calculate_correlations(y_e, y_gt)
    
def prepare_and_correlate_order(x_e, y_e, x_gt, y_gt):
    x_e, y_e, x_gt, y_gt = prepare_data(x_e, y_e, x_gt, y_gt)
    return calculate_correlations_with_ground_truth(y_e, y_gt)
    
def calculate_pearson_stack(s1, s2):
    return stats.pearsonr(s1.ravel(), s2.ravel())
