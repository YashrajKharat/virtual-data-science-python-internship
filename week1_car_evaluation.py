# Week 1 - Car Evaluation Dataset
# Data Acquisition, Cleaning and Exploratory Analysis

# Install once in Google Colab:
# !pip install ucimlrepo

import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# 1. Data Acquisition
car_evaluation = fetch_ucirepo(id=19)

X = car_evaluation.data.features
y = car_evaluation.data.targets
df = pd.concat([X, y], axis=1)

print(df.head())
print("Shape:", df.shape)

# 2. Initial inspection
print(df.info())
print(df.describe(include="all"))

# 3. Missing-value check
print("Missing values:")
print(df.isnull().sum())

# UCI documents this dataset as having no missing values.
# Therefore, no imputation is required.

# 4. Duplicate check
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# 5. Data-type correction
categorical_cols = [
    "buying", "maint", "doors", "persons",
    "lug_boot", "safety", "class"
]
for col in categorical_cols:
    df[col] = df[col].astype("category")

print(df.dtypes)

# 6. Exploratory analysis
print(df["class"].value_counts())

# Visualization 1: target distribution
df["class"].value_counts().plot(kind="bar")
plt.title("Distribution of Car Acceptability Classes")
plt.xlabel("Class")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

# Visualization 2: buying-price distribution
df["buying"].value_counts().plot(kind="bar")
plt.title("Distribution of Buying Price Categories")
plt.xlabel("Buying Price")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

# Visualization 3: missing values
df.isnull().sum().plot(kind="bar")
plt.title("Missing Values by Column")
plt.xlabel("Column")
plt.ylabel("Missing Values")
plt.tight_layout()
plt.show()

# Additional useful checks
print("Unique values:")
for col in df.columns:
    print(col, df[col].unique())

print("\nFinal shape:", df.shape)
