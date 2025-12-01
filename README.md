# aiap-phishing-detection

AIAP22 – End-to-End Machine Learning Pipeline

Full Name: Foo Chit Yan
Email: cyfoo@hotmail.sg

This repository contains my submission for AI Singapore’s AI Apprenticeship Programme (AIAP) – Technical Assessment (Batch 22).
The project implements an end-to-end machine learning pipeline for phishing website detection using the provided SQLite dataset.

1. Repository Structure

	├── .github/                   # GitHub Actions (auto-installs requirements & runs run.sh)
	├── src/
	│   ├── data_loader.py         # Data ingestion from SQLite, feature/target split
	│   ├── preprocessing.py        # Cleaning, imputation, scaling, encoding
	│   ├── models.py              # ML model definitions (LogReg, RandomForest, GradientBoosting)
	│   ├── evaluation.py          # Metrics calculation & pretty printing
	│   ├── main.py                # Main pipeline entry point
	│   ├── __init__.py
	├── phishing.db                # Provided dataset (SQLite)
	├── eda.ipynb                  # Exploratory Data Analysis notebook (Task 1)
	├── run.sh                     # Script executed by GitHub Actions
	├── requirements.txt           # Python dependencies
	└── README.md                  # Project documentation


2. Pipeline Overview (Design & Flow)

   Phishing.db -> Data Ingestion (data_loader.load_data) -> Preprocessing Pipeline (Missing Values, Scaling Numeric) -> Model Training (Gradient Boosting, Logistic Regression,
   Random Forest) -> Model Evaluation (Precision, Recall, F1 Score)

4. Dataset Summary
	•	10,500 rows, 16 features, including both numeric and categorical fields.
	•	Target variable: label (phishing vs legitimate)

  Notable Observations
	•	Highly skewed numeric distributions → required scaling (StandardScaler)
	•	Some features included negative values due to bad data (e.g., NoOfImage = -31) → flagged for cleaning
	•	“Unnamed: 0” identified as an index column → removed before training
	•	Categorical features (Industry, HostingProvider) had missing values → required imputation
	•	Correlation analysis showed strongest signals in:
	•	NoOfSelfRedirect
	•	LargestLineLength
	•	DomainAgeMonths
	•	Robots (binary)

  Outliers
	•	Severe outliers existed in numeric fields
	•	Boxplots guided scaling + model selection

4. Feature Processing Summary

   Feature Type: Numeric, Categorical, Dropped Columns, Target Variable
   Columns: Total 13 columns (Median imputation + StandardScaler)
   Categorical: Industry, HostingProvider (Most frequent imputation)
   Dropped: Unnamed: 0 (Remove the whole column entirely)
   Target Variable: Label (Stratified split)

5. Model Selection

   Logistic Regression: Fast, interpretable baseline
   Performs well on datasets with many linear relationships

   Random Forest: Handles non-linear patterns
   Robust to outliers & missing values
   Works well with mixed feature types

   Gradient Boosting: More powerful sequential tree ensemble
   More powerful sequential tree ensemble
   More powerful sequential tree ensemble

6. Model Evaluation Results

   Random forest and gradient boosting outperform logistic regression significantly.
   Random Forest shows the best recall -> useful in phishing detection
   Gradient boosting shows the best accuracy -> balanced strong performer
   
   



