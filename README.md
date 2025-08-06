#Titanic Survival Prediction with Logistic Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the code and analysis for **LAB 19**, where we build and evaluate a **logistic regression model** to predict survival on the Titanic based on **Age** and **Sex**.

---

## 📌 Overview

The Titanic dataset is a classic in machine learning and statistics education. In this lab, we go beyond simple modeling by focusing on:

- Proper data preprocessing
- Handling missing data (especially `Age`)
- Diagnosing and fixing common modeling errors (e.g., `MissingDataError`)
- Validating logistic regression assumptions
- Improving model fit with splines and interaction terms

We use **statsmodels** for full statistical inference and **Random Forest imputation** to handle missing age values.

---

## 📂 Files

- `lab19_titanic_logistic_regression.ipynb` – Jupyter notebook with full analysis.
- `train.csv` – Dataset from [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data).
- `README.md` – This file.

---

## 🧪 Key Steps Covered

1. **Data Preprocessing**
   - Map `Sex`: `male` → 1, `female` → 0
   - Handle missing `Age` values using **Random Forest Regressor** imputation

2. **Model Building**
   - Logistic regression with `statsmodels.api.Logit`
   - Fix `MissingDataError: exog contains inf or nans` by checking for `NaN` or `inf`

3. **Diagnostics**
   - Variance Inflation Factor (VIF) for multicollinearity
   - Linearity in logit check using scatter plots and Box-Tidwell test
   - Residual analysis and model summary

4. **Model Enhancement**
   - Nonlinear effects via **B-splines** (`patsy.dmatrix`)
   - Interaction between `Sex` and `Age`
   - Visualization of survival probability by age and gender

5. **Visualization**
   - Strip plots, boxplots, histograms
   - Predicted probability curves with confidence-like trends

---


