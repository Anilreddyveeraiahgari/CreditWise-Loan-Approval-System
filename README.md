# CreditWise Loan System

A supervised machine learning project that explores historical loan applications and compares classification models for preliminary loan approval prediction.

## Problem Statement

SecureTrust Bank, a fictional financial company serving urban and rural regions of India, aims to reduce delays and inconsistencies in manual loan application screening. CreditWise explores how applicant financial, employment and credit information can support preliminary approval recommendations before human verification.

## Dataset

The supplied `loan_approval_data.csv` contains 1,000 records and 20 columns, including the applicant identifier and target.

| Feature group | Examples |
|---|---|
| Applicant profile | Age, Gender, Education_Level, Marital_Status, Dependents |
| Employment | Employment_Status, Employer_Category |
| Financial information | Applicant_Income, Coapplicant_Income, Savings, Collateral_Value |
| Credit information | Credit_Score, Existing_Loans, DTI_Ratio |
| Loan details | Loan_Amount, Loan_Term, Loan_Purpose, Property_Area |
| Target | Loan_Approved |

The raw target contains 652 `No`, 298 `Yes`, and 50 missing values. The intended task is binary classification.

## Technology Stack

- Python 3.12 (original notebook: 3.12.4)
- Pandas and NumPy — data processing
- Matplotlib and Seaborn — exploratory data analysis
- Scikit-learn — preprocessing, model training and evaluation
- Jupyter Notebook — experimentation

## Machine Learning Workflow

1. Load and inspect the dataset, including missing values and descriptive statistics.
2. Apply numerical mean imputation and remove `Applicant_ID`.
3. Perform exploratory data analysis using distribution plots, box plots and a correlation heatmap.
4. Apply label encoding and one-hot encoding to categorical variables.
5. Split data into training (80%) and testing (20%) sets using `random_state=42`.
6. Standardize features using `StandardScaler` fitted on the training set.
7. Train and evaluate Logistic Regression, K-Nearest Neighbors (k=5) and Gaussian Naive Bayes.
8. Experiment with squared credit score and debt-to-income ratio features, replacing their original columns.

## Model Performance

The following results are recorded in the supplied notebook. Precision, recall and F1 are weighted averages.

| Model | Feature set | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Original | 79.50% | 73.58% | 79.50% | 76.43% |
| KNN | Original | 67.50% | 61.44% | 67.50% | 63.74% |
| Logistic Regression | Engineered | 79.50% | 73.64% | 79.50% | 76.45% |
| KNN | Engineered | 66.00% | 59.83% | 66.00% | 62.14% |
| Gaussian Naive Bayes | Engineered | 11.00% | 61.59% | 11.00% | 9.06% |

Logistic Regression achieved the highest recorded accuracy and weighted F1-score. Feature engineering produced a small improvement in its weighted precision and F1-score, but did not improve accuracy.

**Important:** These are experimental notebook results, not validated binary-classification performance. Missing target values were encoded as a third class, and preprocessing was fitted before the train-test split. Correcting these issues and reevaluating is necessary before interpreting the results as reliable generalization estimates.

## Project Structure

Recommended repository layout (rename the supplied files as shown):

```text
CreditWise-Loan-System/
├── CREDITWISE_LOAN_SYSTEM.ipynb
├── loan_approval_data.csv
├── CreditWise Loan System.pdf
├── requirements.txt
└── README.md
```

## Installation and Usage

From the project directory, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and launch Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Rename `loan_approval_data(1).csv` to `loan_approval_data.csv` so the notebook can locate it. Open `CREDITWISE_LOAN_SYSTEM.ipynb` (or the original uploaded notebook) and run all cells in order.

## Limitations and Future Improvements

- Remove rows with missing target labels before encoding; explicitly map `No` to 0 and `Yes` to 1.
- Correct categorical imputation (the current notebook applies its second imputer to numerical columns again).
- Use a stratified split and fit all learned preprocessing only on training data, preferably with a Scikit-learn pipeline.
- Apply cross-validation and hyperparameter tuning; evaluate per-class metrics, ROC-AUC and precision-recall performance.
- Assess fairness across demographic groups, review sensitive features and validate against applicable lending requirements.
- Add model serialization, automated tests and an API or interactive interface as future deployment work.

## Responsible Use

This repository is an educational machine learning prototype, not a production lending system. It predicts historical approval labels, not demonstrated repayment risk or creditworthiness. Its outputs should not be used to make actual lending decisions without independent validation, fairness assessment, regulatory review and appropriate human oversight.
