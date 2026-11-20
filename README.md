# Customer Churn Predictor

An end-to-end machine learning project that predicts which telecom customers
are likely to leave, and chooses the decision threshold that minimizes
business cost. Includes an interactive Streamlit app.

## Problem
Losing a customer costs far more than sending a retention offer. A model
that only maximizes accuracy misses most churners, so this project tunes
the decision threshold on a cost model instead of using the default 0.5.

## Dataset
IBM Telco Customer Churn: 7,043 customers, 21 columns, about 26% churn.
Download from Kaggle and save as `data/telco.csv`:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Approach
1. Cleaned the data (blank `TotalCharges` for new customers) and explored churn
   by contract type, tenure, monthly charges and internet service
2. Engineered features: `num_services`, `tenure_bucket`, `charge_ratio`
3. Compared Logistic Regression, Random Forest and XGBoost using
   5-fold stratified cross-validation on the training set only
4. Handled class imbalance with class weights
5. Chose the decision threshold by minimizing an assumed business cost
6. Explained the model with SHAP and analysed the churners it missed
7. Deployed a Streamlit app for single-customer predictions

## Results

Cross-validation (training set):

| Model | ROC-AUC | PR-AUC | F1 | Recall |
|---|---|---|---|---|
| Logistic Regression | [..] | [..] | [..] | [..] |
| Random Forest | [..] | [..] | [..] | [..] |
| XGBoost | [..] | [..] | [..] | [..] |

Held-out test set (XGBoost): **ROC-AUC 0.842, PR-AUC 0.654**

Threshold tuning: the lowest-cost threshold was about **0.17**, versus the
default 0.5. Total cost fell from [Rs ..] to [Rs ..], a saving of [..]%.
The cost curve is fairly flat between 0.1 and 0.3, so the result is not
sensitive to the exact value.

Cost assumptions (placeholders, not real company data):
Rs 15,000 per missed churner, Rs 1,500 per retention offer.

![Cost curve](images/cost_curve.png)

## Key findings
- [Top SHAP feature, e.g. contract type, and which direction it pushes churn]
- [Second and third most important features]
- [What the missed churners have in common, from your error analysis]

![SHAP summary](images/shap_summary.png)

## Streamlit app
Enter a customer's details and get a churn probability with a
High risk / Low risk label based on the tuned threshold.

![High risk example](images/app_high_risk.png)
![Low risk example](images/app_low_risk.png)

## Project structure
```
churn-predictor/
├── data/telco.csv
├── notebooks/main.ipynb     # EDA, model comparison, SHAP
├── features.py              # shared feature engineering
├── train.py                 # trains and saves model.pkl
├── app.py                   # Streamlit app
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Limitations
- The cost values are assumptions. A real company would use its own numbers.
- The threshold was tuned on the held-out test set. In production I would
  tune it on cross-validated predictions from the training data.
- Probabilities are not calibrated because of class weighting, so they are
  best used for ranking customers rather than as literal chances.
- The app fixes some inputs (gender, streaming services, backup) to defaults.
- A single public dataset, so results may not transfer to other companies.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn,
Streamlit
