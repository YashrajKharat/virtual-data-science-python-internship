"""
Week 4 - Machine Learning Model Development and Evaluation
Decision Tree classification using a reproducible car-evaluation-style dataset.

The dataset uses the same six categorical input features and four target labels
used in the UCI Car Evaluation task, but the rows/labels are generated locally
from a transparent scoring rule so the project can run without downloading data.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
from sklearn.inspection import permutation_importance

# 1. Create reproducible categorical dataset
buying = ['vhigh','high','med','low']
maint = ['vhigh','high','med','low']
doors = ['2','3','4','5more']
persons = ['2','4','more']
lug_boot = ['small','med','big']
safety = ['low','med','high']

rows = list(itertools.product(
    buying, maint, doors, persons, lug_boot, safety
))
df = pd.DataFrame(rows, columns=[
    'buying','maint','doors','persons','lug_boot','safety'
])

# Ordinal encoding used only to construct the educational target.
ord_maps = {
    'buying': {'vhigh':0,'high':1,'med':2,'low':3},
    'maint': {'vhigh':0,'high':1,'med':2,'low':3},
    'doors': {'2':0,'3':1,'4':2,'5more':3},
    'persons': {'2':0,'4':1,'more':2},
    'lug_boot': {'small':0,'med':1,'big':2},
    'safety': {'low':0,'med':1,'high':2},
}
weights = {
    'buying':2.2, 'maint':2.0, 'doors':0.25,
    'persons':2.4, 'lug_boot':1.0, 'safety':3.0
}

score = np.zeros(len(df))
for col, weight in weights.items():
    score += df[col].map(ord_maps[col]).to_numpy() * weight

# Hierarchical interactions: capacity/safety and boot/safety matter more.
score += ((df['persons']=='more') & (df['safety']=='high')).astype(float) * 1.4
score += ((df['lug_boot']=='big') & (df['safety']=='high')).astype(float) * 0.8
score += (
    (df['buying'].isin(['low','med'])) &
    (df['maint'].isin(['low','med']))
).astype(float) * 0.5

# Very low capacity or low safety strongly penalizes acceptability.
score -= (
    (df['persons']=='2') | (df['safety']=='low')
).astype(float) * 5.0

df['_score'] = score
df['_row'] = np.arange(len(df))
df = df.sort_values(['_score','_row'], ascending=[False, True]).reset_index(drop=True)

# Reference class proportions used for the reproducible educational replica.
target_counts = {'vgood':65, 'good':69, 'acc':384, 'unacc':1210}
labels = []
for label, count in target_counts.items():
    labels.extend([label] * count)

df['class'] = labels
df = df.sort_values('_row').drop(columns=['_score','_row']).reset_index(drop=True)

# 2. Prepare data
X = df.drop(columns='class')
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 3. Encode categorical variables and train Decision Tree
preprocessor = ColumnTransformer([
    ('categorical', OneHotEncoder(handle_unknown='ignore'), X.columns)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        criterion='entropy',
        max_depth=6,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# 4. Predictions and evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
classes = model.named_steps['classifier'].classes_

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

y_test_bin = label_binarize(y_test, classes=classes)
weighted_auc = roc_auc_score(
    y_test_bin, y_prob, multi_class='ovr', average='weighted'
)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))
print("ROC-AUC  :", round(weighted_auc, 4))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 5. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=classes)
plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.savefig("01_confusion_matrix.png", dpi=180)
plt.show()

# 6. Multiclass ROC curve
plt.figure(figsize=(8, 6))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr,tpr):.3f})")
plt.plot([0,1], [0,1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve - One-vs-Rest")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("02_multiclass_roc_curve.png", dpi=180)
plt.show()

# 7. Permutation feature importance
perm = permutation_importance(
    model, X_test, y_test, n_repeats=10,
    random_state=42, scoring='accuracy'
)
importance = pd.Series(
    perm.importances_mean, index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importance.index, importance.values)
plt.xlabel("Mean decrease in accuracy")
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.savefig("03_feature_importance.png", dpi=180)
plt.show()
