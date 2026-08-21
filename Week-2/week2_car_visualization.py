# Week 2 - Advanced Data Visualization and Storytelling
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Load the same dataset used in Week 1
car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features
y = car_evaluation.data.targets
df = pd.concat([X, y], axis=1)

# 1. Target distribution
df["class"].value_counts().plot(kind="bar")
plt.title("Distribution of Car Acceptability")
plt.xlabel("Class")
plt.ylabel("Records")
plt.tight_layout()
plt.show()

# 2. Feature distributions
for col in df.columns[:-1]:
    df[col].value_counts().plot(kind="bar")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Records")
    plt.tight_layout()
    plt.show()

# 3. Feature entropy / diversity
import numpy as np
def entropy(series):
    p = series.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()

entropy_scores = {
    col: entropy(df[col]) for col in df.columns[:-1]
}
print("Feature entropy:", entropy_scores)

# 4. Target percentage
target_pct = df["class"].value_counts(normalize=True) * 100
target_pct.plot(kind="pie", autopct="%1.1f%%")
plt.title("Share of Car Acceptability Classes")
plt.ylabel("")
plt.show()

# 5. Feature cardinality
cardinality = df.nunique().drop("class")
cardinality.sort_values().plot(kind="barh")
plt.title("Number of Categories per Feature")
plt.xlabel("Number of categories")
plt.show()
