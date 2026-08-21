# Week 3 - Statistical Analysis and Hypothesis Testing
# UCI Car Evaluation Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare

classes = ["unacc", "acc", "good", "vgood"]
observed = np.array([1210, 384, 69, 65])
total = observed.sum()

# H0: The four car acceptability classes are equally distributed.
# H1: The four car acceptability classes are not equally distributed.

expected = np.repeat(total / len(classes), len(classes))

chi2, p_value = chisquare(f_obs=observed, f_exp=expected)

print("Observed:", observed)
print("Expected:", expected)
print("Chi-square statistic:", chi2)
print("Degrees of freedom:", len(classes) - 1)
print("p-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("Reject H0: the class distribution is significantly different from equal.")
else:
    print("Fail to reject H0.")

# Approximate 95% confidence intervals for class proportions
proportions = observed / total
se = np.sqrt(proportions * (1 - proportions) / total)
ci_low = proportions - 1.96 * se
ci_high = proportions + 1.96 * se

results = pd.DataFrame({
    "class": classes,
    "observed": observed,
    "percentage": proportions * 100,
    "ci_low": ci_low * 100,
    "ci_high": ci_high * 100
})
print(results)

plt.bar(classes, observed)
plt.title("Observed Car Acceptability Counts")
plt.xlabel("Class")
plt.ylabel("Number of records")
plt.show()
