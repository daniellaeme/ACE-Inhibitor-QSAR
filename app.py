import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdMolDescriptors

from src.features import smiles_to_morgan
from src.applicability_domain import check_domain

# App Configuration
st.set_page_config(page_title="ACE Inhibitor QSAR")
st.title("ACE Inhibitor Activity Predictor")
st.markdown("""
Welcome to the Angiotensin-Converting Enzyme Quantitative Structure-Activity Relationship deployment
""")


# Asset Loading
@st.cache_resource   # To cache large files
def load_assets():
    xgb_model = joblib.load('models/xgb_tuned_model.pkl')
    app_domain_assets = joblib.load('models/a_d_assets.pkl')

    return xgb_model, app_domain_assets


xgb_model, app_domain_assets = load_assets()


def process_molecule(smiles, xgb_model, app_domain_assets):
    """
    Takes a SMILES string and returns its prediction, confidence and
    Applicability Domain status as a dictionary.
    """
    fp_array = smiles_to_morgan(smiles=smiles, return_as_numpy=True)
    if fp_array is None:
        return None

    # Applicability Domain
    pca_model = app_domain_assets['pca']
    centroid = app_domain_assets['centroid']
    threshold = app_domain_assets['threshold']
    distance, is_in_domain = check_domain(fp_array, pca_model, centroid, threshold)

    # Bioactivity Prediction
    fp_2d = fp_array.reshape(1, -1)

    prediction = xgb_model.predict(fp_2d)[0]
    proba = xgb_model.predict_proba(fp_2d)[0]
    if prediction==1:
        pred_label = "Active"
        confidence = proba[1]
    else:
        pred_label = "Inactive"
        confidence = proba[0]

    return {
        "SMILES": smiles,
        "AD_Distance": float(distance),
        "In_Domain": is_in_domain,
        "Prediction": pred_label,
        "Confidence": confidence,
        "Threshold": threshold,
        "fp_2d": fp_2d
    }

# Sidebar for Input
with st.sidebar:
    st.header("Molecule Input")
    input_mode = st.radio("Choose Input Mode", ["Single SMILES", "CSV Upload"])

smiles_input = None
csv_input = None

# For a single molecule
if input_mode == "Single SMILES":
    smiles_input = st.text_input("Enter SMILES", "CC(=O)Oc1ccccc1C(=O)O")
    st.caption("Example: Aspirin")

    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)

        if mol:
            st.success("Valid SMILES string provided.")
            img = Draw.MolToImage(mol, size=(250, 250))
            st.image(img)
            st.code(smiles_input, language="text")
            st.divider()

            result = process_molecule(smiles_input, xgb_model, app_domain_assets)

            st.subheader("Applicability Domain")
            if not result['In_Domain']:
                st.warning(f'Not within domain! \n(Distance: {result["AD_Distance"]:.2f} > {result["Threshold"]:.2f}).'
                           f' Results might be unreliable.')
            else:
                st.success(f"Within domain! \n(Distance: {result['AD_Distance']:.2f} <= {result['Threshold']:.2f})")

            st.subheader("Bioactivity Prediction")
            if result['Prediction'] == 'Active':
                st.success(f"ACTIVE (Confidence: {result['Confidence']:.2f})")
            else:
                st.info(f"INACTIVE (Confidence: {result['Confidence']:.2f})")

            st.subheader("SHAP")
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer(result["fp_2d"])

            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        else:
            st.error("Invalid SMILES string!")


# For CSV Batch Upload
elif input_mode == "CSV Upload":
    st.subheader("Batch Prediction Mode")
    upload = st.file_uploader("Upload your .csv file")

    if upload:
        df = pd.read_csv(upload)
        st.write("Preview:", df.head())
        col = st.selectbox("Select column:", df.columns)

        if st.button("Screen All"):
            result_list = []
            progress_bar = st.progress(0)
            results = []

            for index,row in df.iterrows():
                smiles = str(row[col])
                result = process_molecule(smiles, xgb_model, app_domain_assets)
                if result is not None:
                    result.pop('fp_2d')
                    result_list.append(result)

                progress_bar.progress((min((index + 1) / len(df), 1.0)))

            result_df = pd.DataFrame(result_list)
            st.dataframe(result_df)
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV file",
                data=csv,
                file_name='batch_predictions.csv',
                mime='text/csv'
            )
