🌟 Loan Default Prediction — Machine Learning Pipeline (Kaggle Project)

Predicting whether applicants will repay their loans using advanced ML models such as LightGBM and K-Fold cross-validation.

📌 Overview

This project builds a full end-to-end machine learning pipeline to predict loan repayment probability based on financial and demographic attributes.

The workflow includes:

Data cleaning & preprocessing

Exploratory Data Analysis (EDA)

Feature engineering

Model training (baseline → advanced models)

Hyperparameter tuning

Performance evaluation using AUC

This project is originally developed in Kaggle Notebook and integrated into GitHub for reproducibility.

📓 Notebook

🔗 Kaggle Notebook (original)
https://www.kaggle.com/code/trishawu/fork-of-predicting-loan-payback

🔗 GitHub Notebook version
/notebooks/fork-of-predicting-loan-payback.ipynb

📁 Project Structure
ml-project/
│
├── notebooks/                 
│     └── fork-of-predicting-loan-payback.ipynb    # main notebook
│
├── README.md                                        # project documentation
└── requirements.txt (optional)

🔍 Exploratory Data Analysis (EDA)

Major insights include:

Credit Score Distribution
Borrowers with higher credit scores show a significantly higher repayment probability.

Income Levels & Loan Default
A clear separation exists between the income distribution of repayers vs. defaulters.

Missing Data Investigation
Missing values in credit_score and potentially high-variance fields were analyzed and handled.

Correlation Heatmap
Identified key high-impact features contributing to model performance (e.g., credit score, annual income).

🛠️ Feature Engineering

The following transformations were applied:

One-hot encoding for categorical variables

Normalization & scaling for numeric fields

Missing value imputation

Removal of low-variance features

Train/validation splits with stratification

Custom feature interactions

🤖 Models Implemented
Baseline Models

Logistic Regression

Decision Tree

Advanced Models

LightGBM (LGBMClassifier)

K-Fold Cross Validation

LightGBM was tuned using grid search over:

num_leaves

colsample_bytree

subsample

reg_lambda

📈 Performance Summary
Best AUC Score (LightGBM)
Hyperparameters	AUC
num_leaves = 16
colsample_bytree = 0.2
subsample = 0.8
reg_lambda = 0.0	0.9249

✔ Improved significantly over baseline logistic regression
✔ Consistent across folds

⚙️ Technology Stack

Python

pandas, numpy

scikit-learn

LightGBM

Matplotlib, Seaborn

Kaggle Notebook

👩‍💻 Author

Trisha (Peixu) Wu
New York, NY
GitHub: https://github.com/peixuwu0

Kaggle: https://www.kaggle.com/trishawu

📬 Contact

Feel free to reach out for collaboration or questions!
