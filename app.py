import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdMolDescriptors

from src.features import generate_morgan_fingerprint
# from src.domain


# App Configuration
st.set_page_config(page_title="ACE Inhibitor QSAR")
st.title("ACE Inhibitor Activity Predictor")
st.markdown("""
Welcome to the Angiotensin-Converting Enzyme Quantitative Structure-Activity Relationship deployment
""")

# Asset Loading
@st.cache_resource   # To cache large files
def load_assets():
    xgb_model = joblib.load("xgb_tuned_model.pkl")
    pca_model = joblib.load('pca')

    return model, pca

xgb_model, pca_model = load_assets()

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

            st.write("Applicability Domain")
            

        else:
            st.error("Invalid SMILES string.")

