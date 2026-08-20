"""
Week 2 - Advanced Data Visualization and Storytelling
Dataset: UCI Car Evaluation

Run from the Week-2 folder:
    python code/week2_visualization.py

The script:
1. Downloads the UCI Car Evaluation dataset if data/car.data is missing.
2. Loads and validates the dataset.
3. Creates six visualizations.
4. Writes a text summary with dataset statistics and key findings.
"""

from pathlib import Path
from urllib.request import urlopen
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "visualizations"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
DATA_FILE = DATA_DIR / "car.data"

COLUMNS = [
    "buying", "maint", "doors", "persons",
    "lug_boot", "safety", "class"
]

# -----------------------------
# Load dataset
# -----------------------------
if not DATA_FILE.exists():
    print("car.data not found. Downloading from UCI...")
    try:
        with urlopen(DATA_URL, timeout=30) as response:
            DATA_FILE.write_bytes(response.read())
        print("Dataset downloaded successfully.")
    except Exception as exc:
        raise SystemExit(
            "Could not download the dataset. "
            "Place your existing car.data file inside Week-2/data/ "
            f"and run again.\nReason: {exc}"
        )

df = pd.read_csv(DATA_FILE, names=COLUMNS)

print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------
# Styling
# -----------------------------
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

# 1. Overall class distribution
plt.figure(figsize=(8, 5))
order = ["unacc", "acc", "good", "vgood"]
ax = sns.countplot(data=df, x="class", order=order)
ax.set_title("Overall Distribution of Car Evaluation Classes")
ax.set_xlabel("Car Evaluation")
ax.set_ylabel("Number of Cars")
save_plot("01_class_distribution.png")

# 2. Safety vs class
plt.figure(figsize=(9, 6))
ax = sns.countplot(data=df, x="safety", hue="class",
                   order=["low", "med", "high"],
                   hue_order=order)
ax.set_title("Safety Level and Car Evaluation")
ax.set_xlabel("Safety Level")
ax.set_ylabel("Number of Cars")
ax.legend(title="Evaluation")
save_plot("02_safety_vs_class.png")

# 3. Buying price vs class
plt.figure(figsize=(9, 6))
ax = sns.countplot(data=df, x="buying", hue="class",
                   order=["low", "med", "high", "vhigh"],
                   hue_order=order)
ax.set_title("Buying Price and Car Evaluation")
ax.set_xlabel("Buying Price")
ax.set_ylabel("Number of Cars")
ax.legend(title="Evaluation")
save_plot("03_buying_vs_class.png")

# 4. Persons vs class heatmap
table = pd.crosstab(df["persons"], df["class"]).reindex(
    index=["2", "4", "more"], columns=order, fill_value=0
)
plt.figure(figsize=(9, 5.5))
sns.heatmap(table, annot=True, fmt="d")
plt.title("Passenger Capacity and Car Evaluation")
plt.xlabel("Car Evaluation")
plt.ylabel("Passenger Capacity")
save_plot("04_persons_vs_class_heatmap.png")

# 5. Buying + safety interaction
pivot = pd.crosstab(
    [df["buying"], df["safety"]],
    df["class"],
    normalize="index"
).reindex(columns=order, fill_value=0)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, fmt=".2f")
plt.title("Evaluation Share by Buying Price and Safety")
plt.xlabel("Car Evaluation")
plt.ylabel("Buying Price / Safety")
save_plot("05_buying_safety_interaction.png")

# 6. Maintenance + safety interaction
pivot2 = pd.crosstab(
    [df["maint"], df["safety"]],
    df["class"],
    normalize="index"
).reindex(columns=order, fill_value=0)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot2, annot=True, fmt=".2f")
plt.title("Evaluation Share by Maintenance Cost and Safety")
plt.xlabel("Car Evaluation")
plt.ylabel("Maintenance / Safety")
save_plot("06_maint_safety_interaction.png")

# -----------------------------
# Quantitative findings
# -----------------------------
class_counts = df["class"].value_counts().reindex(order)
safety_class = pd.crosstab(df["safety"], df["class"], normalize="index").reindex(
    index=["low", "med", "high"], columns=order, fill_value=0
)
buying_class = pd.crosstab(df["buying"], df["class"], normalize="index").reindex(
    index=["low", "med", "high", "vhigh"], columns=order, fill_value=0
)

summary = []
summary.append("WEEK 2 DATA VISUALIZATION SUMMARY")
summary.append("=" * 45)
summary.append(f"Dataset rows: {len(df)}")
summary.append(f"Dataset columns: {len(df.columns)}")
summary.append("Missing values: " + str(int(df.isnull().sum().sum())))
summary.append("")
summary.append("Class counts:")
for k, v in class_counts.items():
    summary.append(f"  {k}: {int(v)}")

summary.append("")
summary.append("Safety-level evaluation proportions:")
for level in safety_class.index:
    best = safety_class.loc[level].idxmax()
    summary.append(
        f"  {level}: highest share = {best} ({safety_class.loc[level, best]:.2%})"
    )

summary.append("")
summary.append("Buying-price evaluation proportions:")
for level in buying_class.index:
    best = buying_class.loc[level].idxmax()
    summary.append(
        f"  {level}: highest share = {best} ({buying_class.loc[level, best]:.2%})"
    )

summary.append("")
summary.append("Important interpretation note:")
summary.append(
    "The dataset is categorical and is derived from a hierarchical decision model. "
    "The visualizations describe associations in the dataset; they should not be "
    "interpreted as causal evidence."
)

(BASE / "visualization_summary.txt").write_text(
    "\n".join(summary), encoding="utf-8"
)

print("\nDone. Six visualizations were saved in:", OUT_DIR)
print("A quantitative summary was saved as:", BASE / "visualization_summary.txt")
