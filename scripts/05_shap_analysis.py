import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

DATA_PATH = "data/processed/processed_food_crisis_data.csv"
PARAMS_PATH = "visuals/tables/xgb_best_params.csv"
OUT_FIG = "visuals/figures"
os.makedirs(OUT_FIG, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.sort_values(["countryiso3", "product", "year", "month"]).reset_index(drop=True)

FEATURES = [
    "avg_price", "price_lag_1", "price_lag_3", "price_lag_6",
    "pct_change_1m", "volatility_ratio_3", "volatility_ratio_6",
    "fao_score", "month"
]
TARGET = "crisis_label"

df_model = df[FEATURES + [TARGET, "year"]].dropna()
X = df_model[FEATURES]
y = df_model[TARGET]

# Split must match 04_baseline_model.py: train 2016-2020, test 2023-2024
train_mask = df_model["year"] <= 2020
test_mask  = df_model["year"] >= 2023
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

scale = (y_train == 0).sum() / (y_train == 1).sum()

# Load best parameters saved by 04_baseline_model.py
best_params = pd.read_csv(PARAMS_PATH).drop(columns=["val_auc_roc", "test_auc_roc"]).iloc[0].to_dict()
best_params["n_estimators"]     = int(best_params["n_estimators"])
best_params["max_depth"]        = int(best_params["max_depth"])
best_params["min_child_weight"] = int(best_params["min_child_weight"])

model = XGBClassifier(**best_params, scale_pos_weight=scale, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print(f"Test AUC-ROC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Beeswarm plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary — Feature Impact on Crisis Prediction")
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/19_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("19_shap_beeswarm.png saved.")

# Bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean Absolute)")
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/20_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("20_shap_bar.png saved.")

# Dependence plot for the most important feature
top_feature = pd.Series(
    np.abs(shap_values).mean(axis=0), index=FEATURES
).idxmax()

plt.figure(figsize=(8, 5))
shap.dependence_plot(top_feature, shap_values, X_test, show=False)
plt.title(f"SHAP Dependence — {top_feature}")
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/21_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"21_shap_dependence.png saved. (Top feature: {top_feature})")

print("\nDone. All SHAP figures saved to visuals/figures.")
