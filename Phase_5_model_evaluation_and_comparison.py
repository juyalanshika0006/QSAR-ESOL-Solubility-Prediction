import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from Phase_4_ML import (
    X_test_desc,
    X_test_fp,
    X_test_comb,
    y_test_desc,
    y_test_fp,
    y_test_comb,
    rf_comb_predictions,
)

# Linear Regression
lr_desc = joblib.load("lr_desc.pkl")
lr_fp = joblib.load("lr_fp.pkl")
lr_comb = joblib.load("lr_comb.pkl")

# Random Forest
rf_desc = joblib.load("rf_desc.pkl")
rf_fp = joblib.load("rf_fp.pkl")
rf_comb = joblib.load("rf_comb.pkl")

# Gradient Boosting
gbr_desc = joblib.load("gbr_desc.pkl")
gbr_fp = joblib.load("gbr_fp.pkl")
gbr_comb = joblib.load("gbr_comb.pkl")

# XGBoost
xgb_desc = joblib.load("xgb_desc.pkl")
xgb_fp = joblib.load("xgb_fp.pkl")
xgb_comb = joblib.load("xgb_comb.pkl")

models = {

    "Linear Regression (Descriptors)": (
        lr_desc,
        X_test_desc,
        y_test_desc
    ),

    "Linear Regression (Fingerprints)": (
        lr_fp,
        X_test_fp,
        y_test_fp
    ),

    "Linear Regression (Combined)": (
        lr_comb,
        X_test_comb,
        y_test_comb
    ),

    "Random Forest (Descriptors)": (
        rf_desc,
        X_test_desc,
        y_test_desc
    ),

    "Random Forest (Fingerprints)": (
        rf_fp,
        X_test_fp,
        y_test_fp
    ),

    "Random Forest (Combined)": (
        rf_comb,
        X_test_comb,
        y_test_comb
    ),

    "Gradient Boosting (Descriptors)": (
        gbr_desc,
        X_test_desc,
        y_test_desc
    ),

    "Gradient Boosting (Fingerprints)": (
        gbr_fp,
        X_test_fp,
        y_test_fp
    ),

    "Gradient Boosting (Combined)": (
        gbr_comb,
        X_test_comb,
        y_test_comb
    ),

    "XGBoost (Descriptors)": (
        xgb_desc,
        X_test_desc,
        y_test_desc
    ),

    "XGBoost (Fingerprints)": (
        xgb_fp,
        X_test_fp,
        y_test_fp
    ),

    "XGBoost (Combined)": (
        xgb_comb,
        X_test_comb,
        y_test_comb
    )
}
results = []

for name, (model, X_test, y_test) in models.items():

    predictions = model.predict(X_test)

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    results.append({
        "Model": name,
        "R2 Score": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")

import pandas as pd

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

results_df.to_csv(
    "Model_Evaluation_Results.csv",
    index=False
)
#Actual vs Predicted plot
plt.figure(figsize=(6,6))

plt.scatter(
    y_test_comb,
    rf_comb_predictions
)

plt.xlabel("Actual Solubility")

plt.ylabel("Predicted Solubility")

plt.title("Actual vs Predicted")

plt.show()

plt.plot(
    [
        y_test_comb.min(),
        y_test_comb.max()
    ],
    [
        y_test_comb.min(),
        y_test_comb.max()
    ],
    color="purple",
)
#Residuals plot
residuals = y_test_comb - rf_comb_predictions
plt.figure(figsize=(6,6))

plt.scatter(
    y_test_comb,
    rf_comb_predictions
)

plt.plot(
    [y_test_comb.min(), y_test_comb.max()],
    [y_test_comb.min(), y_test_comb.max()],
    color="purple"
)

plt.xlabel("Actual Solubility")
plt.ylabel("Predicted Solubility")
plt.title("Actual vs Predicted")

plt.show()

#Error Distribution plot
plt.figure(figsize=(7,5))

plt.hist(
    residuals,
    bins=30
)

plt.xlabel("Residual")

plt.ylabel("Frequency")

plt.title("Residual Distribution")

plt.show()

#Compare models based on R² score

plt.figure(figsize=(12,6))

colors = []

for model in results_df["Model"]:

    if "Linear Regression" in model:
        colors.append("royalblue")

    elif "Random Forest" in model:
        colors.append("forestgreen")

    elif "Gradient Boosting" in model:
        colors.append("orange")

    elif "XGBoost" in model:
        colors.append("crimson")

bars = plt.bar(
    results_df["Model"],
    results_df["R2 Score"],
    color=colors
)

for bar in bars:

    height = bar.get_height()

    if height >= 0:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.03,
            f"{height:.2f}",
            ha="center"
        )
    else:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height - 0.25,
            f"{height:.2f}",
            ha="center"
        )

plt.xticks(rotation=45, ha="right")
plt.ylabel("R² Score")
plt.xlabel("Models")
plt.title("Comparison of Regression Models")
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()