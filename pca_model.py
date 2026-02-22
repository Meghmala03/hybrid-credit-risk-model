# =========================================================
# HYBRID CREDIT RISK MODEL – FULL PIPELINE
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

sns.set(style="whitegrid")

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_excel("insolvency_model_data.xlsx", sheet_name="financial_data")

# =========================================================
# 2. CLEAN NUMERIC COLUMNS
# =========================================================

numeric_cols = [
    "total_assets",
    "short_term_borrowings",
    "long_term_borrowings",
    "revenue",
    "ebit",
    "net_profit",
    "interest_expense",
    "operating_cash_flow"
]

for col in numeric_cols:
    df[col] = (
        df[col].astype(str)
        .str.replace(",", "", regex=False)
        .replace("nan", "0")
        .astype(float)
    )

# =========================================================
# 3. CREATE FINANCIAL RATIOS
# =========================================================

df["total_debt"] = df["short_term_borrowings"] + df["long_term_borrowings"]

df["debt_to_assets"] = np.where(
    df["total_assets"] != 0,
    df["total_debt"] / df["total_assets"],
    0
)

df["interest_coverage"] = np.where(
    df["interest_expense"] != 0,
    df["ebit"] / df["interest_expense"],
    0
)

df["ocf_to_debt"] = np.where(
    df["total_debt"] != 0,
    df["operating_cash_flow"] / df["total_debt"],
    0
)

df["roa"] = np.where(
    df["total_assets"] != 0,
    df["net_profit"] / df["total_assets"],
    0
)

# =========================================================
# 4. PCA
# =========================================================

features = ["debt_to_assets", "interest_coverage", "ocf_to_debt", "roa"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

pca = PCA()
pca.fit(X_scaled)

# =========================================================
# 5. LOGISTIC REGRESSION
# =========================================================

X = df[features]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================================================
# 6. ROC + CONFUSION MATRIX PANEL
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_value = roc_auc_score(y_test, y_prob)

axes[0].plot(fpr, tpr, linewidth=2, color="#6C91BF")
axes[0].plot([0,1],[0,1], linestyle="--", color="gray")
axes[0].set_title(f"ROC Curve (AUC = {auc_value:.2f})")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Default","Default"],
    yticklabels=["No Default","Default"],
    ax=axes[1]
)

axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# =========================================================
# 7. CORRELATION HEATMAP
# =========================================================

plt.figure(figsize=(6,5))
sns.heatmap(
    df[features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Financial Ratio Correlation Matrix")
plt.tight_layout()
plt.show()

# =========================================================
# 8. SCORECARD CONSTRUCTION
# =========================================================

scorecard = X_test.copy()
scorecard["Actual"] = y_test.values
scorecard["PD"] = y_prob

# PD to Score transformation
BaseScore = 650
PDO = 40
BaseOdds = 1/50

B = PDO / np.log(2)
A = BaseScore + B * np.log(BaseOdds)

pd_safe = np.clip(y_prob, 0.001, 0.999)
odds = pd_safe / (1 - pd_safe)
scorecard["Score"] = A - B * np.log(odds)

# Normalize Financial Score
min_s = scorecard["Score"].min()
max_s = scorecard["Score"].max()

scorecard["Financial_Score"] = (
    (scorecard["Score"] - min_s) / (max_s - min_s)
) * 100

# Company Names
scorecard["Company"] = scorecard.index.map(
    lambda i: df.loc[i, "company_name"]
)

scorecard = scorecard.drop_duplicates("Company")

# =========================================================
# 9. QUALITATIVE FACTORS
# =========================================================

industry_map = {
    "VODAFONE": 40,
    "STEEL": 55,
    "CONSULTANCY": 85,
    "ITC": 90
}

scorecard["Industry_Score"] = scorecard["Company"].apply(
    lambda x: next(
        (industry_map[k] for k in industry_map if k in x.upper()),
        65
    )
)

scorecard["Demographic_Score"] = 75
scorecard["Adverse_Media_Score"] = 80
scorecard["Governance_Score"] = 85

# =========================================================
# 10. WEIGHTED HYBRID SCORE (50/10/20/10/10)
# =========================================================

scorecard["Final_Score"] = (
    0.50 * scorecard["Financial_Score"] +
    0.10 * scorecard["Industry_Score"] +
    0.20 * scorecard["Demographic_Score"] +
    0.10 * scorecard["Adverse_Media_Score"] +
    0.10 * scorecard["Governance_Score"]
)

scorecard_sorted = scorecard.sort_values("Final_Score")

# =========================================================
# 11. FINAL HYBRID BAR CHART (PASTEL, CLEAN)
# =========================================================

plt.close("all")

fig, ax = plt.subplots(figsize=(10,6))

pastels = ["#A8DADC","#BFD8B8","#FFD6A5","#E4C1F9"]

bars = ax.bar(
    scorecard_sorted["Company"],
    scorecard_sorted["Final_Score"],
    color=pastels[:len(scorecard_sorted)]
)

for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}",
        xy=(bar.get_x()+bar.get_width()/2, height),
        xytext=(0,6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

ax.set_ylim(0,110)
ax.set_title("5-Factor Hybrid Credit Assessment")
ax.set_ylabel("Composite Credit Score (0 = Weak, 100 = Strong)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# =========================================================
# 12. STACKED FACTOR BREAKDOWN
# =========================================================

factor_cols = [
    "Financial_Score",
    "Industry_Score",
    "Demographic_Score",
    "Adverse_Media_Score",
    "Governance_Score"
]

colors = ["#A8DADC","#BFD8B8","#FFD6A5","#F1C0E8","#CDB4DB"]

scorecard_sorted.set_index("Company")[factor_cols].plot(
    kind="bar",
    stacked=True,
    figsize=(10,6),
    color=colors
)

plt.title("Factor Contribution Breakdown")
plt.ylabel("Score Components")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# =========================================================
# 13. RADAR CHART
# =========================================================

categories = factor_cols
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

for _, row in scorecard_sorted.iterrows():
    values = row[categories].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=row["Company"])
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0,100)
ax.set_title("Radar Comparison of Companies", pad=25)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=False
)
plt.tight_layout()
plt.show()


def risk_bucket(score):
    if score >= 80:
        return "Low Risk"
    elif score >= 60:
        return "Moderate Risk"
    elif score >= 40:
        return "Elevated Risk"
    else:
        return "High Risk"

scorecard["Risk_Bucket"] = scorecard["Final_Score"].apply(risk_bucket)

print(scorecard[["Company","Final_Score","Risk_Bucket"]])

def risk_bucket(score):
    if score >= 80:
        return "Low Risk"
    elif score >= 60:
        return "Moderate Risk"
    elif score >= 40:
        return "Elevated Risk"
    else:
        return "High Risk"

scorecard["Risk_Bucket"] = scorecard["Final_Score"].apply(risk_bucket)

print(scorecard[["Company","Final_Score","Risk_Bucket"]])

bucket_summary = scorecard.groupby("Risk_Bucket")["Company"].count()
print(bucket_summary)

# =========================================================
# 14. FINAL SCORE DISTRIBUTION
# =========================================================

plt.figure(figsize=(6,4))
sns.histplot(scorecard["Final_Score"], kde=True, color="#A8DADC")
plt.title("Distribution of Final Hybrid Scores")
plt.tight_layout()
plt.show()

# =========================================================
# 15. FINANCIAL RATIO HEATMAP
# =========================================================

heatmap_data = df.loc[
    scorecard_sorted.index,
    features
]

heatmap_data.index = scorecard_sorted["Company"]

plt.figure(figsize=(8,5))
sns.heatmap(
    heatmap_data,
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)
plt.title("Financial Ratio Heatmap")
plt.tight_layout()
plt.show()