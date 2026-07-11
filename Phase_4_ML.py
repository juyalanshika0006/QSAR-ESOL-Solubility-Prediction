import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor
import joblib
# Load Processed Dataset
df = pd.read_csv("ESOL_processed.csv")

# Prepare Descriptor Feature Matrix
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

# Combined Features
X_combined = pd.concat(
    [X_desc, fingerprint_df],
    axis=1
)

# Target Variable
y = df["Solubility"]
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
#Train linear regression models 
lr_desc = LinearRegression()

lr_desc.fit(
    X_train_desc,
    y_train_desc
)

lr_desc_predictions = lr_desc.predict(
    X_test_desc
)
#fingerprints
lr_fp = LinearRegression()

lr_fp.fit(
    X_train_fp,
    y_train_fp
)

lr_fp_predictions = lr_fp.predict(
    X_test_fp
)
#combined features
lr_comb = LinearRegression()

lr_comb.fit(
    X_train_comb,
    y_train_comb
)

lr_comb_predictions = lr_comb.predict(
    X_test_comb
)

#Train Random Forest Regressor models
#for descriptor features
rf_desc = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_desc.fit(
    X_train_desc,
    y_train_desc
)

rf_desc_predictions = rf_desc.predict(
    X_test_desc
)
#for fingerprinnts
rf_fp = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_fp.fit(
    X_train_fp,
    y_train_fp
)

rf_fp_predictions = rf_fp.predict(
    X_test_fp
)
#for combined features
rf_comb = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_comb.fit(
    X_train_comb,
    y_train_comb
)

rf_comb_predictions = rf_comb.predict(
    X_test_comb
)
#Train Gradient Boosting Regressor models
#for descriptor features
gbr_desc = GradientBoostingRegressor(
    random_state=42
)

gbr_desc.fit(
    X_train_desc,
    y_train_desc
)

gbr_desc_predictions = gbr_desc.predict(
    X_test_desc
)
#for fingerprints
gbr_fp = GradientBoostingRegressor(
    random_state=42
)

gbr_fp.fit(
    X_train_fp,
    y_train_fp
)

gbr_fp_predictions = gbr_fp.predict(
    X_test_fp
)
#for combined features
gbr_comb = GradientBoostingRegressor(
    random_state=42
)

gbr_comb.fit(
    X_train_comb,
    y_train_comb
)

gbr_comb_predictions = gbr_comb.predict(
    X_test_comb
)
#Train XGBoost Regressor models
#for descriptor features
xgb_desc = XGBRegressor(
    random_state=42,
    n_estimators=100
)

xgb_desc.fit(
    X_train_desc,
    y_train_desc
)

xgb_desc_predictions = xgb_desc.predict(
    X_test_desc
)
#for fingerprints
xgb_fp = XGBRegressor(
    random_state=42,
    n_estimators=100
)

xgb_fp.fit(
    X_train_fp,
    y_train_fp
)

xgb_fp_predictions = xgb_fp.predict(
    X_test_fp
)
#for combined features
xgb_comb = XGBRegressor(
    random_state=42,
    n_estimators=100
)

xgb_comb.fit(
    X_train_comb,
    y_train_comb
)

xgb_comb_predictions = xgb_comb.predict(
    X_test_comb
)
#combing all predictions into a single DataFrame for each model
xgb_model=pd.DataFrame({
    "XGB_Combined": xgb_comb_predictions,
    "XGB_Fingerprints": xgb_fp_predictions,
    "XGB_Descriptors": xgb_desc_predictions
})
gb_model=pd.DataFrame({
    "GBR_Combined": gbr_comb_predictions,
    "GBR_Fingerprints": gbr_fp_predictions,
    "GBR_Descriptors": gbr_desc_predictions
})
rf_model=pd.DataFrame({
    "RF_Combined": rf_comb_predictions, 
    "RF_Fingerprints": rf_fp_predictions,
    "RF_Descriptors": rf_desc_predictions
})
linear_regressor=pd.DataFrame({
    "LR_Combined": lr_comb_predictions,
    "LR_Fingerprints": lr_fp_predictions,
    "LR_Descriptors": lr_desc_predictions
})

#storing all predictions in a dictionary
predictions = {
    "LR_Descriptors": lr_desc_predictions,
    "LR_Fingerprints": lr_fp_predictions,
    "LR_Combined": lr_comb_predictions,

    "RF_Descriptors": rf_desc_predictions,
    "RF_Fingerprints": rf_fp_predictions,
    "RF_Combined": rf_comb_predictions,

    "GBR_Descriptors": gbr_desc_predictions,
    "GBR_Fingerprints": gbr_fp_predictions,
    "GBR_Combined": gbr_comb_predictions,

    "XGB_Descriptors": xgb_desc_predictions,
    "XGB_Fingerprints": xgb_fp_predictions,
    "XGB_Combined": xgb_comb_predictions
}
#verifying the predictions
print("Models trained successfully!")

print()

print("Total models:", len(predictions))

# Save Trained Models

# Linear Regression Models
joblib.dump(
    lr_desc,
    "lr_desc.pkl"
)

joblib.dump(
    lr_fp,
    "lr_fp.pkl"
)

joblib.dump(
    lr_comb,
    "lr_comb.pkl"
)

# Random Forest Models
joblib.dump(
    rf_desc,
    "rf_desc.pkl"
)

joblib.dump(
    rf_fp,
    "rf_fp.pkl"
)

joblib.dump(
    rf_comb,
    "rf_comb.pkl"
)

# Gradient Boosting Models
joblib.dump(
    gbr_desc,
    "gbr_desc.pkl"
)

joblib.dump(
    gbr_fp,
    "gbr_fp.pkl"
)

joblib.dump(
    gbr_comb,
    "gbr_comb.pkl"
)

# XGBoost Models
joblib.dump(
    xgb_desc,
    "xgb_desc.pkl"
)

joblib.dump(
    xgb_fp,
    "xgb_fp.pkl"
)

joblib.dump(
    xgb_comb,
    "xgb_comb.pkl"
)

print()
print("All trained models have been saved successfully!")
