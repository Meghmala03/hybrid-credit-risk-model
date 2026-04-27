# Hybrid Credit Risk Model (Insolvency Prediction)

This project develops a hybrid credit risk framework to estimate Probability of Default (PD) for listed Indian firms.

The focus is not just prediction, but building a system that:

ranks firms by risk
produces interpretable scores
reflects real-world credit risk thinking

# Model Framework
The model combines statistical, financial, and qualitative components:

1. Statistical Model
Logistic Regression on WoE-transformed variables
Outputs 1-year PD
2. Financial Risk Component

# Decomposed into:
Leverage Risk → Debt to Assets
Liquidity Risk → Interest Coverage, Cash Flow
3. Industry Risk Overlay

# Firms mapped to industry-level risk categories:
Telecom → High
Steel → Medium
Engineering → Low
IT / FMCG → Very Low
4. Hybrid PD
PD_final = 0.4 × Statistical PD
         + 0.4 × Financial PD
         + 0.2 × Industry Risk
5. Rule-Based PD (Business Overlay)

# To mimic practitioner-style credit assessment:
Leverage capped at 2%
Liquidity capped at 1%
Industry risk added separately
Total PD capped at 3%
6. Lifetime PD
PD_5Y = 1 - (1 - PD)^5

# Model Evaluation
A. Discrimination
AUC ≈ 0.62
Gini ≈ 0.24
KS ≈ 0.50
B. Class Imbalance
Default rate ≈ 20%
Moderate imbalance
Class weighting tested but avoided due to instability
C. Backtesting (Key Insight)

# Firms grouped into risk buckets:
Higher PD → higher default rate
Lower PD → minimal defaults
Which Confirms monotonic risk ordering

# Outputs:
Probability of Default (PD)
Credit Score (0–100 scale)
Risk Buckets

# Visualizations:
Risk ranking
Backtesting (monotonicity)
ROC curve

# Key Takeaways
High AUC does not guarantee meaningful risk ranking
Classification metrics can be unstable in small datasets
Monotonicity is critical for model validation
Credit models are more useful for ranking than classification

 # Tech Stack
Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
