import matplotlib.pyplot as plt
from visturing.properties import prop5

def calculate_diffs(img1, img2):
    return ((img1-img2).mean(axis=(1,2,3))**2)**(1/2)

output = prop5.evaluate(calculate_diffs,
                                     data_path='../Data/Experiment_5/',
                                     gt_path='../Data/ground_truth/')
print(output)
plt.figure()
for d in output["diffs"]:
    plt.plot(d)
plt.show()
