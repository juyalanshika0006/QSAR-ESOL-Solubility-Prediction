import joblib
from Phase_4_ML import (X_train_comb, y_train_comb,X_test_comb)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor

#Tuning hyperparameters for Linear Regression

#parameter grid
lr_params = {
    "fit_intercept":[True,False]
}

lr_grid = GridSearchCV(

    estimator=LinearRegression(),

    param_grid=lr_params,

    cv=5,

    scoring="r2",

    n_jobs=-1
)
print("Starting Linear Regression...")
lr_grid.fit(
    X_train_comb,
    y_train_comb
)

#Best parameters for Linear Regression
print(lr_grid.best_params_)
#Best model for Linear Regression
lr_model=lr_grid.best_estimator_
print("Linear Model has been trained")

#Tuning hyperparameters for Random Forest

#parameter grid
rf_params = {

    "n_estimators":[100,200,300],

    "max_depth":[None,10,20,30],

    "min_samples_split":[2,5,10],

    "min_samples_leaf":[1,2,4],

    "max_features":["sqrt","log2"]
}
#grid search for Random Forest
rf_grid = GridSearchCV(

    RandomForestRegressor(
        random_state=42
    ),

    rf_params,

    cv=5,

    scoring="r2",

    n_jobs=-1
)

rf_grid.fit(
    X_train_comb,
    y_train_comb
)

#best models
best_rf = rf_grid.best_estimator_

print("Random Forest Model has been trained")

#Tuning hyperparameters for Gradient Boosting
#parameter grid
gbr_params = {

    "n_estimators":[100,200],

    "learning_rate":[0.01,0.05,0.1],

    "max_depth":[3,4,5],

    "subsample":[0.8,1.0]
}
#grid search for Gradient Boosting
gbr_grid = GridSearchCV(

    GradientBoostingRegressor(
        random_state=42
    ),

    gbr_params,

    cv=5,

    scoring="r2",

    n_jobs=-1
)
gbr_grid.fit(
    X_train_comb,
    y_train_comb
)
#best models
best_gbr = gbr_grid.best_estimator_
print("Gradient Boosting Model has been trained")

#Tuning hyperparameters for XGBoost
#parameter grid
xgb_params = {

    "n_estimators":[100,200,300],

    "learning_rate":[0.01,0.05,0.1],

    "max_depth":[3,5,7],

    "subsample":[0.8,1.0],

    "colsample_bytree":[0.8,1.0]
}

#grid search for XGBoost
xgb_grid = GridSearchCV(

    XGBRegressor(
        random_state=42
    ),

    xgb_params,

    cv=5,

    scoring="r2",

    n_jobs=-1
)

xgb_grid.fit(
    X_train_comb,
    y_train_comb
)    
#best models
best_xgb = xgb_grid.best_estimator_
print("XGBoost Model has been trained")

print(lr_model)

print(best_rf)

print(best_gbr)

print(best_xgb)

print(lr_model)

print(rf_grid.best_score_)

print(gbr_grid.best_score_)

print(xgb_grid.best_score_)


rf_best_predictions = best_rf.predict(
    X_test_comb
)
print(">>> Reached the saving section")

joblib.dump(
    best_rf,
    "best_random_forest.pkl"
)

joblib.dump(
    best_gbr,
    "best_gradient_boosting.pkl"
)

joblib.dump(
    best_xgb,
    "best_xgboost.pkl"
)

best_parameters = {

    "Random Forest":rf_grid.best_params_,

    "Gradient Boosting":gbr_grid.best_params_,

    "XGBoost":xgb_grid.best_params_,

    "Linear Regression":lr_grid.best_params_
}