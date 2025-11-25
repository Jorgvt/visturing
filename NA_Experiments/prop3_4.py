from visturing.properties import prop3_4

def calculate_diffs(img1, img2):
    return ((img1-img2).mean(axis=(1,2,3))**2)**(1/2)

diffs, pearson, order = prop3_4.evaluate(calculate_diffs,
                                     data_path='../Data/Experiment_4_5/',
                                     gt_path='../Data/ground_truth/')
print("Pearson correlation: ", pearson)
print("Order: ", order)
