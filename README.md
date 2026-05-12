**Predicting Antihypertensive Bioactivity: A QSAR Model for Human Angiotensin-Converting Enzyme**

*Project Overview*
This repository contains an end-to-end Machine Learning pipeline that predicts the biological activity of chemical compounds against the Angiotensin-converting enzyme (ACE), a primary target for blood pressure medications.

By simulating the early-stage virtual screening process used in pharmaceutical drug discovery, this project translates raw chemical structures into machine-readable Morgan Fingerprints and classifies them as active or inactive inhibitors.

*Data Source*
The raw bioactivity data (IC50 values) was mined programmatically from the open-access ChEMBL Database (Target ID: CHEMBL1808 - Homo sapiens). The data was strictly filtered for binding assays to ensure biological relevance.

*Tech Stack*
Data Acquisition: ChEMBL API (chembl_webresource_client)

Data Manipulation: pandas, numpy

Chemoinformatics: rdkit (SMILES processing, Lipinski descriptors, Morgan Fingerprints)

Machine Learning: scikit-learn (Random Forest, XGBoost)

Deployment: streamlit

*Project Structure*
00_Data_Acquisition.ipynb: API mining and initial filtering.
ace_raw.csv: The uncleaned, raw dataset.

01_EDA.ipynb: Handling missing values and binary classification setup.

02_Data_Splitting_and_Model_Selection.ipynb: traditional 80-20 split vs  and training baseline classification models.

03_Model_Validation_and_Applicability_Domain.ipynb: Handling missing values and binary classification setup.

04_SHAP.ipynb: Handling missing values and binary classification setup.

features.py: Functions to convert smiles to Morgan fingerprints, etc.

model_evaluator.py: Functions to calculate f1 score, precision. recall, accuracy and confusion matrix metrics.

scaffold_split.py: Functions handling splitting the datasets into scaffolds.

applicability_domain.py: Functions to draw the applicability domain.

app.py: The streamlit app.

a_d_assets.pkl: The pca model.

xgb_tuned_model.pkl: The final tuned model.

