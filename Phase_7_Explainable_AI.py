import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap

#Loading the best models
best_xgb_model = joblib.load('best_xgboost.pkl')

#Loading the test data
df=pd.read_csv('ESOL_processed.csv')

#Recreating the features and target variable
descriptor_columns = [
    "MolecularWeight",
    "LogP",
    "Polar Surface Area",
    "Number of H-Bond Donors",
    "Number of Rotatable Bond",
    "Number of Rings",
]

X_desc = df[descriptor_columns]

# Prepare Fingerprint Feature Matrix
fingerprint_df = df[[str(i) for i in range(1024)]]


X_combined = pd.concat(
    [X_desc, fingerprint_df],
    axis=1
)

y = df["Solubility"]

#Train/Test Split
X_train_desc, X_test_desc, y_train_desc, y_test_desc = train_test_split(
    X_desc,
    y,
    test_size=0.2,
    random_state=42
)
X_train_fp, X_test_fp, y_train_fp, y_test_fp = train_test_split(
    fingerprint_df,
    y,
    test_size=0.2,
    random_state=42
)
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(
    X_combined,
    y,
    test_size=0.2,
    random_state=42
)


#Predict
predictions=best_xgb_model.predict(X_test_comb)

#Feature Importance using Permuation Importance
importance = best_xgb_model.feature_importances_

#Creating a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X_combined.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

#Plot the feature importance 
top_features=feature_importance_df.head(20)
plt.figure(figsize=(10,8))
plt.barh(
    top_features["Feature"].astype(str),
    top_features["Importance"],
    color="pink"
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 Feature Importance (XGBoost)",
          fontweight="bold")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()
plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)   

#Plot permuatation importance

print("Starting permutation importance...")

perm_importance = permutation_importance(
    best_xgb_model,
    X_test_comb,
    y_test_comb,
    n_repeats=3,          # reduce from 10
    random_state=42,
    scoring="r2",
    n_jobs=-1             # use all CPU cores
)
perm_df = pd.DataFrame({
    "Feature": X_test_comb.columns,
    "Importance": perm_importance.importances_mean
})

# Sort by importance
perm_df = perm_df.sort_values(
    by="Importance",
    ascending=False
)


top20 = perm_df.head(20)
print("Plotting.........")
plt.figure(figsize=(12,8))

plt.barh(
    top20["Feature"],
    top20["Importance"],
    color="steelblue"
)

plt.gca().invert_yaxis()

plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title(
    "Top 20 Permutation Feature Importance (XGBoost)",
    fontsize=16,
    fontweight="bold"
)

plt.grid(axis="x", linestyle="--", alpha=0.4)

for i, v in enumerate(top20["Importance"]):
    plt.text(v + 0.001, i, f"{v:.3f}", va="center")

plt.tight_layout()

plt.savefig(
    "permutation_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#shap explainer
explainer=shap.TreeExplainer(best_xgb_model)
#calculate shap values
shap_values=explainer(X_test_comb)
print("Shap summary plot forming")
#Shap summary plot
plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_test_comb,
    show=False
)

plt.title(
    "SHAP Summary Plot (XGBoost)",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#shap dependence plot
#for LogP
plt.figure(figsize=(8,6))

shap.plots.scatter(
    shap_values[:, "LogP"],
    show=False
)

plt.title(
    "SHAP Dependence Plot - LogP",
    fontsize=15,
    fontweight="bold"
)

plt.savefig(
    "shap_dependence_logp.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


#For Mol.Wt.

plt.figure(figsize=(8,6))

shap.plots.scatter(
    shap_values[:, "MolecularWeight"],
    show=False
)

plt.title(
    "SHAP Dependence Plot - MolecularWeight",
    fontsize=15,
    fontweight="bold"
)

plt.savefig(
    "shap_dependence_MolecularWeight.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#For Polar surface area

plt.figure(figsize=(8,6))

shap.plots.scatter(
    shap_values[:, "Polar Surface Area"],
    show=False
)

plt.title(
    "SHAP Dependence Plot - Polar Surface Area",
    fontsize=15,
    fontweight="bold"
)

plt.savefig(
    "shap_dependence_Polar Surface Area.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#Waterfallplot of SHAP

shap_values = explainer(X_test_comb)

# Plot for the first molecule
shap.plots.waterfall(
    shap_values[0],
    max_display=20,
    show=False
)

plt.savefig(
    "shap_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()