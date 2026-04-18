##Predicting Antihypertensive Bioactivity: A QSAR Model for Human Angiotensin-Converting Enzyme##

#Project Overview#
This repository contains an end-to-end Machine Learning pipeline that predicts the biological activity of chemical compounds against the Angiotensin-converting enzyme (ACE), a primary target for blood pressure medications.

By simulating the early-stage virtual screening process used in pharmaceutical drug discovery, this project translates raw chemical structures into machine-readable Morgan Fingerprints and classifies them as active or inactive inhibitors.

#Data Source#
The raw bioactivity data (IC50 values) was mined programmatically from the open-access ChEMBL Database (Target ID: CHEMBL1808 - Homo sapiens). The data was strictly filtered for binding assays to ensure biological relevance.

#Tech Stack#
Data Acquisition: ChEMBL API (chembl_webresource_client)

Data Manipulation: pandas, numpy

Chemoinformatics: rdkit (SMILES processing, Lipinski descriptors, Morgan Fingerprints)

Machine Learning: scikit-learn (Random Forest, SVM)

Deployment: streamlit

#Project Structure#
01_Data_Acquisition.ipynb: API mining and initial filtering.

02_EDA.ipynb: Handling missing values and binary classification setup.

03_Model_Building.ipynb: Machine Learning and training baseline classification models.

ace_raw.csv: The uncleaned, raw dataset.

(More to be added as the project progresses)