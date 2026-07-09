# 🧪 QSAR-ESOL-Solubility-Prediction

An end-to-end QSAR (Quantitative Structure–Activity Relationship) machine learning project for predicting the aqueous solubility (logS) of small molecules using RDKit, molecular descriptors, Morgan fingerprints, and multiple regression algorithms.

---

## 📖 Project Overview

Aqueous solubility is one of the most important physicochemical properties in drug discovery because it directly affects drug absorption, bioavailability, and formulation.

This project demonstrates a complete cheminformatics and machine learning workflow using the ESOL dataset. Molecular structures are processed with RDKit, transformed into numerical features, and used to train multiple regression models for solubility prediction.

---

## 🎯 Objectives

- Explore and understand the ESOL dataset
- Process molecular structures using RDKit
- Calculate molecular descriptors
- Generate Morgan fingerprints
- Perform exploratory visualization
- Train multiple regression models
- Compare different molecular representations
- Evaluate predictive performance

---

## 📂 Dataset

**Dataset:** ESOL (Delaney)

**Target Variable:**

Measured Log Solubility (logS)

**Number of Molecules:**

1128

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- RDKit
- Scikit-learn
- XGBoost

---

## 📁 Project Structure

```text
QSAR-ESOL-Solubility-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_visualization.ipynb
│   └── 04_machine_learning.ipynb
│
├── figures/
├── models/
├── README.md
└── requirements.txt
```

---

## 🔬 Workflow

```text
ESOL Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
RDKit Molecule Processing
      │
      ▼
Descriptor Calculation
      │
      ▼
Morgan Fingerprint Generation
      │
      ▼
Feature Engineering
      │
      ▼
Exploratory Visualization
      │
      ▼
Machine Learning Models
      │
      ▼
Performance Evaluation
```

---

## 🤖 Machine Learning Models

The following regression models were implemented:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Each model was trained using three different feature representations:

- Molecular Descriptors
- Morgan Fingerprints
- Combined Descriptors + Morgan Fingerprints

This results in **12 independent machine learning experiments** for comparative analysis.

---

## 📊 Exploratory Visualization

Visualizations include:

- Solubility Distribution
- Molecular Weight Distribution
- LogP Distribution
- TPSA Distribution
- Correlation Heatmap
- Descriptor Relationship Scatter Plots
- Pairwise Descriptor Analysis

---

## 📈 Current Progress

- ✅ Phase 1 – Exploratory Data Analysis
- ✅ Phase 2 – RDKit Processing & Feature Engineering
- ✅ Phase 3 – Exploratory Visualization
- ✅ Phase 4 – Machine Learning Model Development
- ⏳ Phase 5 – Model Evaluation & Comparison
- ⏳ Phase 6 – Hyperparameter Tuning
- ⏳ Phase 7 – Explainable AI (Feature Importance & SHAP)
- ⏳ Phase 8 – Final Report & Documentation

---

## 📚 Skills Demonstrated

- Cheminformatics
- RDKit
- Molecular Descriptor Calculation
- Morgan Fingerprints
- Feature Engineering
- Exploratory Data Analysis
- Data Visualization
- Regression Modeling
- Random Forest
- Gradient Boosting
- XGBoost
- Drug Discovery Machine Learning

---

## 🚀 Future Improvements

- Hyperparameter optimization using GridSearchCV
- Cross-validation
- SHAP explainability
- Model serialization
- Predicting solubility for new molecules
- Interactive visualization dashboard

---

## 👩‍💻 Author

**Anshika Juyal**

B.Sc. Biotechnology

Interested in AI for Drug Discovery, Cheminformatics, and Computational Biology.
