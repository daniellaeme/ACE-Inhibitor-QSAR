import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    """
    Converts the SMILES string into a Mogan fingerprint as a numpy array.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintsasBitVect(mmol, radius, n_bits)
    fingerprint_array = np.zeroz((0,), dtype=int)
    DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)

    return fingerprint_array

def get_tanimoto_matrix(fingerprint_list):
    """
    Takes a list of fingerprints and returns an NxN similarity matrix.
    """
    n = len(fingerprint_list)
    similarity_matrix = []
    for i in range(n):
        current_fingerprint = fingerprint_list[i]
        similarities = DataStructs.BulkTanimotoSimilarity(current_fingerprint, fingerprint_list)
        similarity_matrix.append(similarities)

    return np.array(similarity_matrix)
