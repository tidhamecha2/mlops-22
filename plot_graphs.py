# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

from utils import preprocess_digits, train_dev_test_split, h_param_tuning, data_viz, pred_image_viz


train_frac, dev_frac, test_frac = 0.8, 0.1 , 0.1
assert train_frac + dev_frac + test_frac == 1.

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

h_param_comb = [{"gamma": g, "C": c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list) * len(c_list)

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits


x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac
)

# PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()
# define the evaluation metric
metric=metrics.accuracy_score


best_model, best_metric, best_h_params = h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric)

# PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(x_test)

pred_image_viz(x_test, predicted)

# 4. report the test set accurancy with that best model.
# PART: Compute evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

print("Best hyperparameters were:")
print(best_h_params)
