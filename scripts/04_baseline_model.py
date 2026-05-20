import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, average_precision_score,
                             precision_recall_fscore_support,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/processed/processed_food_crisis_data.csv"
OUT_FIG = "visuals/figures"
OUT_TBL = "visuals/tables"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TBL, exist_ok=True)

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

# ── Temporal Train / Val / Test Split ─────────────────────────────────────────
# Data spans 9 years (2016-2024).
# Test set is fixed to the most recent 2 years (2023-2024).
# The remaining 7 years (2016-2022) are split ~5/2 into train/val so that
# val and test are equal in length (2 years each) for comparable evaluation.
#
#   Train : 2016-2020  →  5 years  (~55%)  — model learns from this data
#   Val   : 2021-2022  →  2 years  (~22%)  — model selection and threshold tuning
#   Test  : 2023-2024  →  2 years  (~22%)  — touched only once for the final report

train_mask = df_model["year"] <= 2020
val_mask   = (df_model["year"] >= 2021) & (df_model["year"] <= 2022)
test_mask  = df_model["year"] >= 2023

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

n_total = len(X)
print(f"Train : {len(X_train):>6}  ({len(X_train)/n_total*100:.1f}%)  crisis rate: {y_train.mean():.3f}")
print(f"Val   : {len(X_val):>6}  ({len(X_val)/n_total*100:.1f}%)  crisis rate: {y_val.mean():.3f}")
print(f"Test  : {len(X_test):>6}  ({len(X_test)/n_total*100:.1f}%)  crisis rate: {y_test.mean():.3f}")


def evaluate(name, y_true, y_pred, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"AUC-ROC : {auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    return {"model": name, "auc_roc": auc, "pr_auc": pr_auc,
            "precision": p, "recall": r, "f1": f1}


# ── Baseline Models — evaluated on validation set ────────────────────────────
scale = (y_train == 0).sum() / (y_train == 1).sum()

lr = LogisticRegression(class_weight="balanced", max_iter=1000, solver="saga", random_state=42)
lr.fit(X_train, y_train)
lr_res = evaluate("Logistic Regression", y_val, lr.predict(X_val), lr.predict_proba(X_val)[:, 1])

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_res = evaluate("Random Forest", y_val, rf.predict(X_val), rf.predict_proba(X_val)[:, 1])

xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                    scale_pos_weight=scale, random_state=42, n_jobs=-1, eval_metric="auc")
xgb.fit(X_train, y_train)
xgb_res = evaluate("XGBoost", y_val, xgb.predict(X_val), xgb.predict_proba(X_val)[:, 1])

lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced",
                      random_state=42, n_jobs=-1, verbose=-1)
lgbm.fit(X_train, y_train)
lgbm_res = evaluate("LightGBM", y_val, lgbm.predict(X_val), lgbm.predict_proba(X_val)[:, 1])

# --- Feature importance (Logistic Regression — absolute coefficients) ---
fi_lr = pd.Series(np.abs(lr.coef_[0]), index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
fi_lr.plot(kind="barh", ax=ax, color="mediumpurple")
ax.set_title("Logistic Regression — Feature Importance (|coefficient|)")
ax.set_xlabel("|Coefficient|")
plt.tight_layout()
fig.savefig(f"{OUT_FIG}/08_feature_importance_lr.png", dpi=150)
plt.close()

# --- Feature importance (Random Forest) ---
fi_rf = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
fi_rf.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Random Forest — Feature Importance")
ax.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(f"{OUT_FIG}/09_feature_importance_rf.png", dpi=150)
plt.close()

# --- Feature importance (XGBoost baseline) ---
fi_xgb = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
fi_xgb.plot(kind="barh", ax=ax, color="coral")
ax.set_title("XGBoost — Feature Importance")
ax.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(f"{OUT_FIG}/10_feature_importance_xgb.png", dpi=150)
plt.close()

# --- Feature importance (LightGBM) ---
fi_lgbm = pd.Series(lgbm.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
fi_lgbm.plot(kind="barh", ax=ax, color="mediumseagreen")
ax.set_title("LightGBM — Feature Importance")
ax.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(f"{OUT_FIG}/11_feature_importance_lgbm.png", dpi=150)
plt.close()

# --- Feature importance comparison (all 5 models) ---
# Plotted after tuning so all five are available; built here for layout clarity.
# LR uses |coefficient|; tree models use native feature_importances_.
def _plot_combined_fi(models, out_path):
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    for ax, (fi, color, title, xlabel) in zip(axes, models):
        fi.sort_values(ascending=True).plot(kind="barh", ax=ax, color=color)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    fig.suptitle("Feature Importance Comparison Across Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# ── XGBoost Hyperparameter Tuning ─────────────────────────────────────────────
# TimeSeriesSplit runs only on train data; val and test are never touched here.
param_dist = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3]
}

tscv = TimeSeriesSplit(n_splits=5)
xgb_base = XGBClassifier(scale_pos_weight=scale, random_state=42, n_jobs=-1, eval_metric="auc")

search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_dist,
    n_iter=40,
    scoring="roc_auc",
    cv=tscv,
    random_state=42,
    verbose=2,
    n_jobs=-1
)

print("\nStarting XGBoost hyperparameter tuning...")
search.fit(X_train, y_train)
print(f"Best parameters : {search.best_params_}")
print(f"CV AUC-ROC      : {search.best_score_:.4f}")

best_model = search.best_estimator_
y_prob_tuned_val = best_model.predict_proba(X_val)[:, 1]
y_pred_tuned_val = best_model.predict(X_val)
xgb_tuned_res = evaluate("XGBoost (tuned)", y_val, y_pred_tuned_val, y_prob_tuned_val)

# --- Feature importance (tuned XGBoost) ---
fi_tuned = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
fi_tuned.plot(kind="barh", ax=ax, color="darkorange")
ax.set_title("Tuned XGBoost — Feature Importance")
ax.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(f"{OUT_FIG}/12_feature_importance_xgb_tuned.png", dpi=150)
plt.close()

# --- Combined feature importance: RF vs LightGBM vs Tuned XGBoost ---
_plot_combined_fi(
    [
        (fi_lr,    "mediumpurple",   "Logistic Regression", "|Coefficient|"),
        (fi_rf,    "steelblue",      "Random Forest",       "Importance"),
        (fi_xgb,   "coral",          "XGBoost",             "Importance"),
        (fi_lgbm,  "mediumseagreen", "LightGBM",            "Importance"),
        (fi_tuned, "darkorange",     "XGBoost (tuned)",     "Importance"),
    ],
    f"{OUT_FIG}/13_feature_importance_comparison.png"
)

# ── Threshold Optimisation — determined on val set ───────────────────────────
# Threshold is selected on val; applied to test to prevent leakage.
precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob_tuned_val)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx]
print(f"\n=== Threshold Tuning (on validation set) ===")
print(f"Optimal threshold : {optimal_threshold:.3f}")
print(f"Precision: {precisions[best_idx]:.4f} | Recall: {recalls[best_idx]:.4f} | F1: {f1_scores[best_idx]:.4f}")

# ── Validation Comparison Table & Plots ──────────────────────────────────────
results_val = pd.DataFrame([lr_res, rf_res, xgb_res, lgbm_res, xgb_tuned_res])
results_val = results_val.round(4)
print("\n=== MODEL COMPARISON (Validation) ===")
print(results_val.to_string(index=False))
results_val.to_csv(f"{OUT_TBL}/model_comparison_val.csv", index=False)

colors = ["#d9534f", "#5bc0de", "#f0ad4e", "#5cb85c", "#9b59b6"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(results_val["model"], results_val["auc_roc"], color=colors)
axes[0].set_title("AUC-ROC Comparison (Validation)")
axes[0].set_ylabel("AUC-ROC")
axes[0].set_ylim(0.4, 1.0)
axes[0].tick_params(axis='x', rotation=15)

axes[1].bar(results_val["model"], results_val["pr_auc"], color=colors)
axes[1].set_title("PR-AUC Comparison (Validation)")
axes[1].set_ylabel("PR-AUC")
axes[1].set_ylim(0, results_val["pr_auc"].max() * 1.2)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
fig.savefig(f"{OUT_FIG}/14_model_comparison_bar.png", dpi=150)
plt.close()

# ROC curves (validation — for model selection)
roc_models_val = [
    ("Logistic Regression", lr.predict_proba(X_val)[:, 1],   "#d9534f"),
    ("Random Forest",       rf.predict_proba(X_val)[:, 1],   "#5bc0de"),
    ("XGBoost",             xgb.predict_proba(X_val)[:, 1],  "#f0ad4e"),
    ("LightGBM",            lgbm.predict_proba(X_val)[:, 1], "#5cb85c"),
    ("XGBoost (tuned)",     y_prob_tuned_val,                 "#9b59b6"),
]

fig, ax = plt.subplots(figsize=(8, 6))
for name, y_prob, color in roc_models_val:
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc = roc_auc_score(y_val, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)

ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Model Comparison (Validation)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
fig.savefig(f"{OUT_FIG}/15_roc_curves_val.png", dpi=150)
plt.close()

# ── Final Evaluation — TEST (touched only once) ───────────────────────────────
print("\n" + "=" * 60)
print("FINAL TEST — Best model: XGBoost (tuned)")
print("=" * 60)

y_prob_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= optimal_threshold).astype(int)
test_res = evaluate("XGBoost (tuned) — TEST", y_test, y_pred_test, y_prob_test)

# Confusion matrix (test set, optimal threshold applied)
cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Crisis", "Crisis"], yticklabels=["No Crisis", "Crisis"])
ax.set_title(f"XGBoost (tuned) — Test Confusion Matrix\n(threshold={optimal_threshold:.2f})")
plt.tight_layout()
fig.savefig(f"{OUT_FIG}/16_confusion_matrix_xgb.png", dpi=150)
plt.close()

# ROC curve (test set — final report)
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="#9b59b6", linewidth=2,
        label=f"XGBoost tuned (AUC={test_res['auc_roc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Final Test Set")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(f"{OUT_FIG}/17_roc_curves.png", dpi=150)
plt.close()

# Save best parameters + val and test AUC (read by 06_shap_analysis.py)
best_params_df = pd.DataFrame([search.best_params_])
best_params_df["val_auc_roc"]  = xgb_tuned_res["auc_roc"]
best_params_df["test_auc_roc"] = test_res["auc_roc"]
best_params_df.to_csv(f"{OUT_TBL}/xgb_best_params.csv", index=False)

# Final summary table (val + test result side by side)
test_summary = pd.DataFrame([{
    "model":               "XGBoost (tuned)",
    "optimal_threshold":   round(float(optimal_threshold), 4),
    "val_auc_roc":         round(xgb_tuned_res["auc_roc"], 4),
    "val_pr_auc":          round(xgb_tuned_res["pr_auc"], 4),
    "val_f1":              round(xgb_tuned_res["f1"], 4),
    "test_auc_roc":        round(test_res["auc_roc"], 4),
    "test_pr_auc":         round(test_res["pr_auc"], 4),
    "test_f1":             round(test_res["f1"], 4),
}])
test_summary.to_csv(f"{OUT_TBL}/final_test_result.csv", index=False)

print("\nDone. All metrics and figures saved.")

# ── Validation Length Comparison: 1 Year vs 2 Years ──────────────────────────
# Proof: how well does each val set represent the test set?
# Method: train with the same XGBoost parameters under both configurations,
#         then compare the gap between val_auc and test_auc.
# Smaller gap → val more accurately predicts test performance → more reliable.

print("\n" + "=" * 60)
print("VALIDATION LENGTH COMPARISON: 1 YEAR vs 2 YEARS")
print("=" * 60)

configs = {
    "Val = 1 year\n(2022)":    {"train_end": 2021, "val_start": 2022, "val_end": 2022},
    "Val = 2 years\n(2021-22)": {"train_end": 2020, "val_start": 2021, "val_end": 2022},
}

gap_results = []

for label, cfg in configs.items():
    tr_mask = df_model["year"] <= cfg["train_end"]
    vl_mask = (df_model["year"] >= cfg["val_start"]) & (df_model["year"] <= cfg["val_end"])
    te_mask = df_model["year"] >= 2023

    Xtr, ytr = X[tr_mask], y[tr_mask]
    Xvl, yvl = X[vl_mask], y[vl_mask]
    Xte, yte = X[te_mask], y[te_mask]

    sc = (ytr == 0).sum() / (ytr == 1).sum()
    params = {k: v for k, v in search.best_params_.items()}
    params["n_estimators"]     = int(params["n_estimators"])
    params["max_depth"]        = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])

    m = XGBClassifier(**params, scale_pos_weight=sc, random_state=42,
                      n_jobs=-1, eval_metric="auc")
    m.fit(Xtr, ytr)

    val_auc  = roc_auc_score(yvl, m.predict_proba(Xvl)[:, 1])
    test_auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    gap      = abs(val_auc - test_auc)

    gap_results.append({
        "Configuration": label,
        "Train size":    int(tr_mask.sum()),
        "Val size":      int(vl_mask.sum()),
        "Val AUC-ROC":   round(val_auc,  4),
        "Test AUC-ROC":  round(test_auc, 4),
        "Gap":           round(gap,      4),
    })
    print(f"\n{label.replace(chr(10), ' ')}")
    print(f"  Train: {tr_mask.sum()} | Val: {vl_mask.sum()} | Test: {te_mask.sum()}")
    print(f"  Val AUC : {val_auc:.4f}  |  Test AUC : {test_auc:.4f}  |  Gap : {gap:.4f}")

gap_df = pd.DataFrame(gap_results)
gap_df.to_csv(f"{OUT_TBL}/val_length_comparison.csv", index=False)

# Plot: val vs test AUC side by side + gap
labels_clean = ["Val=1 year\n(2022)", "Val=2 years\n(2021-22)"]
val_aucs  = [r["Val AUC-ROC"]  for r in gap_results]
test_aucs = [r["Test AUC-ROC"] for r in gap_results]
gaps      = [r["Gap"]          for r in gap_results]

x = np.arange(len(labels_clean))
width = 0.3

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: val vs test AUC side by side
bars1 = axes[0].bar(x - width/2, val_aucs,  width, label="Val AUC",  color="#5bc0de")
bars2 = axes[0].bar(x + width/2, test_aucs, width, label="Test AUC", color="#9b59b6")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels_clean)
axes[0].set_ylabel("AUC-ROC")
axes[0].set_title("Val vs Test AUC-ROC")
axes[0].set_ylim(0.4, 1.0)
axes[0].legend()
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

# Right: gap (smaller = more reliable)
bar_colors = ["#e74c3c" if g == max(gaps) else "#2ecc71" for g in gaps]
gap_bars = axes[1].bar(labels_clean, gaps, color=bar_colors, width=0.4)
axes[1].set_ylabel("|Val AUC − Test AUC|")
axes[1].set_title("Val-Test Gap (smaller = more reliable)")
axes[1].set_ylim(0, max(gaps) * 1.5)
for bar, g in zip(gap_bars, gaps):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{g:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.suptitle("Validation Length Comparison: Which Val Set Better Represents the Test Set?",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT_FIG}/18_val_length_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n16_val_length_comparison.png saved.")
print("\nDone. Validation length comparison complete.")
