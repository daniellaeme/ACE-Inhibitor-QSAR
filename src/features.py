import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def smiles_to_morgan(smiles, radius=2, n_bits=2048, return_as_numpy=True):
    """
    Converts the SMILES string into a Mogan fingerprint.
    If return_as_numpy is True, returns a numpy array (for ML model).
    If return_as_numpy is False, returns an RDKit ExplicitBitVect (for Tanimoto similarity).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintasBitVect(mol, radius, n_bits)

    if not return_as_numpy:
        return fingerprint

    fingerprint_array = np.zero((0,), dtype=int)
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
