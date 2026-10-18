# 🧪 QSAR ESOL Solubility Prediction Using Machine Learning

A complete end-to-end **Quantitative Structure–Activity Relationship (QSAR)** project that predicts the aqueous solubility (ESOL) of chemical compounds using molecular descriptors, Morgan fingerprints, and multiple machine learning algorithms.

This project demonstrates the complete drug discovery machine learning workflow—from raw molecular data to model interpretation and virtual screening.

---

## 📌 Project Overview

The objective of this project is to build an accurate machine learning model capable of predicting molecular solubility from SMILES strings.

The workflow includes:

- Data preprocessing
- Molecular descriptor generation
- Molecular fingerprint generation
- Exploratory Data Analysis (EDA)
- Machine Learning model development
- Hyperparameter tuning
- Explainable AI (SHAP)
- Virtual screening pipeline

---

## 🧬 Dataset

Dataset used:

- **Delaney ESOL Dataset**
- Contains experimentally measured aqueous solubility values.
- Molecular structures represented as SMILES strings.

Target Variable:

- **Measured Log Solubility (logS)**

---
## 📈 Model Performance & Results

The models were evaluated using R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). Below is the comparison of baseline models across different feature sets (Descriptors, Morgan Fingerprints, and Combined).

| Rank | Model | Feature Set | R² Score |
| :--- | :--- | :--- | :--- |
| 1 | Gradient Boosting | Combined (Descriptors + FP) | **0.89** |
| 2 | XGBoost | Combined (Descriptors + FP) | **0.88** |
| 3 | Gradient Boosting | Descriptors Only | 0.87 |
| 4 | Random Forest | Combined (Descriptors + FP) | 0.87 |
| 5 | Random Forest | Descriptors Only | 0.86 |
| 6 | XGBoost | Descriptors Only | 0.86 |
| 7 | Linear Regression | Descriptors Only | 0.75 |
| 8 | XGBoost | Fingerprints Only | 0.72 |
| 9 | Random Forest | Fingerprints Only | 0.71 |
| 10 | Gradient Boosting | Fingerprints Only | 0.66 |
| 11 | Linear Regression | Combined (Descriptors + FP) | -2.52 |
| 12 | Linear Regression | Fingerprints Only | -4.26 |

**Key Takeaways:**
* **Feature Engineering Impact:** Tree-based models (Gradient Boosting, XGBoost, Random Forest) performed exceptionally well with both Descriptors and Combined features. 
* **Curse of Dimensionality:** Linear Regression failed when presented with high-dimensional Morgan Fingerprints (R² < 0), proving the necessity of non-linear models for QSAR tasks.
* **Final Selection:** After baseline evaluation, **XGBoost** was selected for hyperparameter tuning (Phase 6) due to its high performance and efficiency, becoming the final model deployed in the virtual screening pipeline.

_ _ _
# 📂 Project Structure

```
QSAR_ESOL_Project/

│
├── Phase_1_Data_exploration.py
├── Phase_2_Data_Cleaning_and_RDKit_Processing.py
├── Phase_3_Exploratory_Visualization.py
├── Phase_4_ML.py
├── Phase_5_model_evaluation_and_comparison.py
├── Phase_6_Hypertuning_Tuning_and_Model_Optimisation.py
├── Phase_7_Explainable_AI.py
├── Phase_8_Virtual_screening_and_pipeline_prediction.py
│
├── ESOL_processed.csv
├── delaney-processed.csv
│
├── *.pkl
├── *.png
├── predicted_solubility.csv
│
└── README.md
```

---

# ⚙️ Technologies Used

- Python
- RDKit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Joblib

---

# 🧪 Phase 1 — Data Exploration

Performed:

- Loaded the ESOL dataset
- Dataset inspection
- Missing value analysis
- Duplicate detection
- Statistical summary
- Target distribution analysis

Outputs:

- Dataset statistics
- Clean dataset overview

---

# 🧪 Phase 2 — Molecular Feature Engineering

Generated molecular descriptors using RDKit:

- Molecular Weight
- LogP
- TPSA
- Hydrogen Bond Donors
- Rotatable Bonds
- Ring Count

Generated Morgan Fingerprints:

- Radius = 2
- Fingerprint Size = 1024 bits

Combined descriptors and fingerprints into a machine learning feature matrix.

---

# 📊 Phase 3 — Exploratory Data Analysis

Visualizations created:

- Solubility distribution
- Molecular Weight distribution
- LogP distribution
- Correlation Heatmap
- Pairplot
- Descriptor relationships

Purpose:

- Understand feature distributions
- Detect feature correlations
- Explore molecular property relationships

---

# 🤖 Phase 4 — Machine Learning Models

Implemented multiple regression models:

### Linear Regression

- Descriptor Model
- Fingerprint Model
- Combined Features Model

### Random Forest Regressor

- Descriptor Model
- Fingerprint Model
- Combined Features Model

### Gradient Boosting Regressor

- Descriptor Model
- Fingerprint Model
- Combined Features Model

### XGBoost Regressor

- Descriptor Model
- Fingerprint Model
- Combined Features Model

Models were saved using Joblib for future prediction.

---

# 📈 Phase 5 — Model Evaluation

Evaluation metrics:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Visualization:

- Actual vs Predicted Plot
- Residual Plot
- Residual Distribution
- Model Comparison

---

# ⚡ Phase 6 — Hyperparameter Optimization

Performed hyperparameter tuning using:

- GridSearchCV

Optimized:

- Random Forest
- Gradient Boosting
- XGBoost

Compared tuned models against baseline models.

Best performing model selected and saved.

---

# 🔍 Phase 7 — Explainable AI (XAI)

Used SHAP (SHapley Additive Explanations) to interpret model predictions.

Generated:

- SHAP Summary Plot
- SHAP Waterfall Plot
- SHAP Dependence Plot
- Permutation Importance
- Feature Importance

Purpose:

- Understand global feature importance
- Explain individual predictions
- Improve model interpretability

---

# 🚀 Phase 8 — Virtual Screening Pipeline

Built a prediction pipeline capable of:

- Reading new molecules from CSV
- Validating SMILES strings
- Calculating RDKit descriptors
- Generating Morgan fingerprints
- Loading trained XGBoost model
- Predicting solubility
- Saving predictions to CSV

Pipeline handles invalid SMILES gracefully.

Output:

```
predicted_solubility.csv
```

---

## 🏆 Final Model Selection

While Gradient Boosting achieved the highest baseline R² score (0.89), **XGBoost** was selected as the final model for deployment. 

Following hyperparameter optimization using **GridSearchCV (Phase 6)**, the tuned XGBoost model demonstrated the best generalization and was saved (`best_xgboost.pkl`) for use in the Virtual Screening Pipeline.
---

# 🧠 Machine Learning Workflow

```
SMILES
   │
   ▼
RDKit
   │
   ▼
Descriptors + Morgan Fingerprints
   │
   ▼
Feature Matrix
   │
   ▼
Machine Learning Models
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Explainable AI (SHAP)
   │
   ▼
Virtual Screening Pipeline
```

---

# 💡 Key Skills Demonstrated

- QSAR Modeling
- Molecular Feature Engineering
- RDKit
- Data Cleaning
- Exploratory Data Analysis
- Machine Learning
- Model Evaluation
- Hyperparameter Optimization
- Explainable AI
- SHAP
- XGBoost
- Virtual Screening
- Drug Discovery Workflow

---

# 📚 Future Improvements

Planned future enhancements include:

- Streamlit Web Application
- Batch Molecule Prediction
- Molecular Structure Visualization
- Model Deployment
- Docker Support
- Cloud Deployment
- REST API
- CI/CD Integration

---

# 👩‍💻 Author

**Anshika Juyal**

B.Sc. Biotechnology

Interested in:

- Drug Discovery
- Bioinformatics
- Machine Learning
- Computational Biology
- AI in Healthcare

---

# ⭐ Acknowledgements

- RDKit
- Scikit-learn
- XGBoost
- SHAP
- Delaney ESOL Dataset
