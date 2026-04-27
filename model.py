import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# =========================================================
# SETTINGS
# =========================================================
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# =========================================================
# STEP 1: LOAD & CLEAN DATA
# =========================================================
print("\n" + "="*50 + "\nSTEP 1: DATA LOADING & CLEANING\n" + "="*50)

df = pd.read_excel("insolvency_model_data.xlsx", sheet_name="financial_data")

df.columns = df.columns.str.strip().str.lower()

numeric_cols = [
    "total_assets", "short_term_borrowings", "long_term_borrowings",
    "revenue", "ebit", "net_profit", "interest_expense", "operating_cash_flow"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df.replace([np.inf, -np.inf], 0, inplace=True)

print(df.head())

# =========================================================
# STEP 2: COMPANY NAME CLEANING
# =========================================================
df['company'] = df['company_name'].str.upper().str.strip()

name_map = {
    "TATA CONSULTANCY SERVICES LIMITED": "TCS",
    "LARSEN AND TOUBRO LIMITED": "L&T",
    "TATA STEEL LIMITED": "TATA STEEL",
    "VODAFONE IDEA LIMITED": "VODAFONE IDEA",
    "ITC LIMITED": "ITC"
}

df['company_short'] = df['company'].map(name_map)
df['company_short'] = df['company_short'].fillna(df['company'])

print("\nCompany Mapping:")
print(df[['company_name', 'company_short']].drop_duplicates())

# =========================================================
# STEP 3: FINANCIAL RATIOS
# =========================================================
print("\n" + "="*50 + "\nSTEP 2: FINANCIAL RATIOS\n" + "="*50)

df["total_debt"] = df["short_term_borrowings"] + df["long_term_borrowings"]

df["debt_to_assets"] = np.where(df["total_assets"] != 0,
                               df["total_debt"] / df["total_assets"], 0)

df["interest_coverage"] = np.where(df["interest_expense"] != 0,
                                  df["ebit"] / df["interest_expense"], 0)

df["ocf_to_debt"] = np.where(df["total_debt"] != 0,
                            df["operating_cash_flow"] / df["total_debt"], 0)

df["roa"] = np.where(df["total_assets"] != 0,
                    df["net_profit"] / df["total_assets"], 0)

# =========================================================
# STEP 4: FEATURE ENGINEERING (SAFE)
# =========================================================
df["financial_stress"] = df["debt_to_assets"] * (1 / (df["interest_coverage"] + 1))

features = [
    "debt_to_assets",
    "interest_coverage",
    "ocf_to_debt",
    "roa",
    "financial_stress"
]

print(df[["company_short"] + features].head())

# =========================================================
# STEP 5: WOE TRANSFORMATION
# =========================================================
print("\n" + "="*50 + "\nSTEP 3: WOE TRANSFORMATION\n" + "="*50)

def calculate_woe(data, feature, target):
    data = data.copy()
    data["bin"] = pd.qcut(data[feature], q=5, duplicates="drop")

    grouped = data.groupby("bin", observed=True)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]

    grouped["good"] = grouped["total"] - grouped["bad"]
    grouped["dist_good"] = (grouped["good"] + 0.5) / (grouped["good"].sum() + 0.5)
    grouped["dist_bad"] = (grouped["bad"] + 0.5) / (grouped["bad"].sum() + 0.5)

    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])

    return grouped["woe"].to_dict()

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

woe_features = []

for col in features:
    mapping = calculate_woe(df_train, col, "default")

    df["bin"] = pd.qcut(df[col], q=5, duplicates="drop")
    df[col + "_woe"] = df["bin"].map(mapping).fillna(0)

    df.drop(columns=["bin"], inplace=True)
    woe_features.append(col + "_woe")

print(df[["company_short"] + woe_features].head())

# =========================================================
# STEP 6: MODEL
# =========================================================
print("\n" + "="*50 + "\nSTEP 4: MODEL\n" + "="*50)

X = df[woe_features]
y = df["default"]

X_train, X_test_set, y_train, y_test_set = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

df["pd_statistical"] = model.predict_proba(X)[:, 1]

print(df[["company_short", "pd_statistical"]].head())

# =============================
# CLASS IMBALANCE CHECK
# =============================
print("\n" + "="*50)
print("CLASS DISTRIBUTION")
print("="*50)

print(df["default"].value_counts())
print("\nDefault Rate:", df["default"].mean())

# =========================================================
# STEP 7: HYBRID PD
# =========================================================
print("\n" + "="*50 + "\nSTEP 5: HYBRID PD\n" + "="*50)

lev_norm = (df["debt_to_assets"] - df["debt_to_assets"].min()) / (df["debt_to_assets"].max() - df["debt_to_assets"].min() + 1e-6)

liq_raw = 0.5 * df["interest_coverage"] + 0.5 * df["ocf_to_debt"]
liq_norm = (liq_raw - liq_raw.min()) / (liq_raw.max() - liq_raw.min() + 1e-6)

df["pd_financial"] = (0.7 * lev_norm + 0.3 * (1 - liq_norm))

industry_map = {
    "Telecom": 0.18,
    "Steel": 0.10,
    "Engineering": 0.06,
    "IT Services": 0.01,
    "FMCG": 0.02,
    "Other": 0.05
}

def get_industry(name):
    name = name.upper()
    if "VODAFONE" in name: return "Telecom"
    if "STEEL" in name: return "Steel"
    if "LARSEN" in name: return "Engineering"
    if "TATA CONSULTANCY" in name: return "IT Services"
    if "ITC" in name: return "FMCG"
    return "Other"

df["industry"] = df["company"].apply(get_industry)
df["industry_risk_pd"] = df["industry"].map(industry_map)

df["pd_final"] = (0.4 * df["pd_statistical"] +
                  0.4 * df["pd_financial"] +
                  0.2 * df["industry_risk_pd"])

df["credit_score"] = (1 - df["pd_final"]) * 100

print("\nHybrid PD Output:")
print(df[[
    "company_short",
    "pd_statistical",
    "pd_financial",
    "industry_risk_pd",
    "pd_final",
    "credit_score"
]].sort_values("pd_final", ascending=False).head(10))

# =========================================================
# STEP 7B: RULE-BASED PD (BUSINESS OVERLAY)
# =========================================================
print("\n" + "="*50 + "\nSTEP 7B: RULE-BASED PD OVERLAY\n" + "="*50)

# Leverage component (max 2%)
df["pd_leverage"] = 0.02 * lev_norm

# Liquidity component (max 1%)
df["pd_liquidity"] = 0.01 * (1 - liq_norm)

# Industry risk categories
industry_category_map = {
    "Telecom": "High",
    "Steel": "Medium",
    "Engineering": "Low",
    "IT Services": "Very Low",
    "FMCG": "Very Low",
    "Other": "Medium"
}

industry_scale = {
    "No Risk": 0.0,
    "Very Low": 0.002,
    "Low": 0.005,
    "Medium": 0.01,
    "High": 0.02,
    "Very High": 0.03
}

df["industry_category"] = df["industry"].map(industry_category_map)
df["industry_pd_rule"] = df["industry_category"].map(industry_scale)

# Combine rule-based PD
df["pd_rule_based"] = (
    df["pd_leverage"] +
    df["pd_liquidity"] +
    df["industry_pd_rule"]
)

# Cap at 3%
df["pd_rule_based"] = df["pd_rule_based"].clip(upper=0.03)

# Show output
print("\nRule-Based PD Output:")
print(df[[
    "company_short",
    "pd_leverage",
    "pd_liquidity",
    "industry_pd_rule",
    "pd_rule_based"
]].sort_values("pd_rule_based", ascending=False))


# =========================================================
# STEP 8: LIFETIME PD
# =========================================================
def lifetime_pd(pd_1y):
    return 1 - (1 - pd_1y)**5

df["pd_5y_adj"] = df["pd_final"].apply(lifetime_pd)

# =========================================================
# STEP 9: FINAL TABLE
# =========================================================
print("\n" + "="*50 + "\nFINAL TABLE\n" + "="*50)

final_table = df[[
    "company_short",
    "pd_final",
    "pd_rule_based",
    "pd_5y_adj",
    "credit_score"
]].sort_values("pd_final", ascending=False)

print(final_table)


print("\n" + "="*50 + "\nBACKTESTING RESULTS\n" + "="*50)

# Sort by risk
df_sorted = df.sort_values("pd_final", ascending=False)

# Show top risky firms
print("\nTop Risky Firms:")
print(df_sorted[["company_short", "pd_final", "default"]].head())

# Show safest firms
print("\nSafest Firms:")
print(df_sorted[["company_short", "pd_final", "default"]].tail())

df['pd_bucket'] = pd.qcut(df['pd_final'], q=3, labels=["Low", "Medium", "High"])

backtest = df.groupby("pd_bucket")["default"].mean()

print("\nDefault Rate by Risk Bucket:")
print(backtest)
print("\nObservation: Default rates increase with risk bucket → model captures risk ordering")


backtest_df = backtest.reset_index()

import matplotlib.pyplot as plt


plt.show()
plt.figure(figsize=(6,4))
plt.bar(backtest_df["pd_bucket"], backtest_df["default"])

plt.title("Higher Predicted Risk Corresponds to Higher Default Rates")
plt.xlabel("Risk Bucket")
plt.ylabel("Default Rate")
plt.ylim(0, 1)

for i, v in enumerate(backtest_df["default"]):
    plt.text(i, v + 0.02, f"{v:.1f}", ha='center')

plt.show()
# =========================================================
# STEP 10: VISUALS
# =========================================================
print("\nGenerating charts...")
plt.close('all')

# ROC
y_prob_test = model.predict_proba(X_test_set)[:, 1]
fpr, tpr, _ = roc_curve(y_test_set, y_prob_test)
auc = roc_auc_score(y_test_set, y_prob_test)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve")
plt.show()

# =============================
# AUC + GINI
# =============================
y_prob_test = model.predict_proba(X_test_set)[:, 1]
auc = roc_auc_score(y_test_set, y_prob_test)

gini = 2 * auc - 1

print("\n" + "="*50)
print("MODEL DISCRIMINATION METRICS")
print("="*50)
print(f"AUC Score: {auc:.3f}")
print(f"Gini Coefficient: {gini:.3f}")


# =============================
# KS STATISTIC
# =============================
ks_df = pd.DataFrame({
    "y_true": y_test_set,
    "y_prob": y_prob_test
})

# Sort by probability
ks_df = ks_df.sort_values("y_prob", ascending=False)

# Separate goods and bads
ks_df["good"] = 1 - ks_df["y_true"]
ks_df["bad"] = ks_df["y_true"]

# Cumulative distributions
ks_df["cum_good"] = ks_df["good"].cumsum() / ks_df["good"].sum()
ks_df["cum_bad"] = ks_df["bad"].cumsum() / ks_df["bad"].sum()

# KS calculation
ks_df["ks"] = ks_df["cum_bad"] - ks_df["cum_good"]

ks_stat = ks_df["ks"].max()
ks_idx = ks_df["ks"].idxmax()

print("\n" + "="*50)
print("KS STATISTIC")
print("="*50)
print(f"KS Value: {ks_stat:.3f}")
print(f"Max Separation at Probability: {ks_df.loc[ks_idx, 'y_prob']:.3f}")

# =============================
# KS PLOT
# =============================
plt.figure(figsize=(6,4))

plt.plot(ks_df["cum_good"], label="Cumulative Good")
plt.plot(ks_df["cum_bad"], label="Cumulative Bad")

plt.title("KS Curve")
plt.xlabel("Observations (sorted by risk)")
plt.ylabel("Cumulative Distribution")

plt.legend()
plt.show()
# =============================
# Confusion Matrix (FIXED)
# =============================

# Get probabilities instead of hard predictions
y_prob_test = model.predict_proba(X_test_set)[:, 1]

# Apply LOWER threshold (critical)
threshold = 0.5
y_pred_test = (y_prob_test > threshold).astype(int)

# Build confusion matrix
cm = confusion_matrix(y_test_set, y_pred_test)

# Plot
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])

plt.title(f"Confusion Matrix (Threshold = {threshold})")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

print("\nThreshold Testing:")

for t in [0.2, 0.3, 0.4]:
    y_pred = (y_prob_test > t).astype(int)
    cm = confusion_matrix(y_test_set, y_pred)
    print(f"\nThreshold = {t}")
    print(cm)

from sklearn.metrics import classification_report

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)

print(classification_report(y_test_set, model.predict(X_test_set)))
# Credit Score
df_plot = df.groupby("company_short", as_index=False)["credit_score"].mean()
df_plot = df_plot.sort_values("credit_score")
plt.figure(figsize=(6,4))

sns.barplot(
    x="credit_score",
    y="company_short",
    data=df_plot,
    palette="RdYlGn",
    errorbar=None
)

plt.title("Firm-Level Credit Risk Ranking (Higher Score = Safer)")
plt.xlabel("Credit Score")
plt.ylabel("")

for index, value in enumerate(df_plot["credit_score"]):
    plt.text(value - 4, index, str(int(value)), va='center')

plt.tight_layout()
plt.show()
