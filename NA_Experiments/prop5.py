from visturing.properties import prop5

def calculate_diffs(img1, img2):
    return ((img1-img2).mean(axis=(1,2,3))**2)**(1/2)

diffs, pearson, order = prop5.evaluate(calculate_diffs,
                                     data_path='../Data/Experiment_3/',
                                     gt_path='../Data/ground_truth/')
print("Pearson correlation: ", pearson)
print("Order: ", order)
