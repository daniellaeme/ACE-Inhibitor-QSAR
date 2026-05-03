import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def smiles_to_morgan(smiles, return_as_numpy=True):
    """
    Converts the SMILES string into a Mogan fingerprint.
    If return_as_numpy is True, returns a numpy array (for ML model).
    If return_as_numpy is False, returns an RDKit ExplicitBitVect (for Tanimoto similarity).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if return_as_numpy:
        return mfpgen.GetFingerprintAsNumpy(mol)
    else:
        return mfpgen.GetFingerprint(mol)


def get_tanimoto_matrix(fingerprint_list):
    """
    Takes a list of fingerprints and returns an NxN similarity matrix.
    The fingerprints passed here must have been generated with return_as_numpy=False.
    """
    n = len(fingerprint_list)
    similarity_matrix = []
    for i in range(n):
        current_fingerprint = fingerprint_list[i]
        similarities = DataStructs.BulkTanimotoSimilarity(current_fingerprint, fingerprint_list)
        similarity_matrix.append(similarities)

    return np.array(similarity_matrix)


def visualise_morgan_bit(smiles_list, bit_id, radius=2, n_bits=2048):
    """
    Searches a list of SMILES for a molecule that activates a specific Morgan bit.
    Returns an image highlighting the substructure.
    For SHAP analysis.
    """
    local_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        info = {}
        # Get the fingerprint and populate the info dictionary with bit mappings
        # We use the classic method specifically for visualization because
        # DrawMorganBit natively expects a plain Python dictionary format.
        _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=info)

        # Check if the target SHAP bit was activated in this specific molecule
        if bit_id in info:
            print(f"Found Bit {bit_id} in SMILES: {smiles}")
            # Draw the molecule, highlighting the atoms responsible for the bit
            img = Draw.DrawMorganBit(mol, bit_id, info)
            return img

    print(f"Could not find a molecule with Bit {bit_id} in the provided list.")
    return None
    