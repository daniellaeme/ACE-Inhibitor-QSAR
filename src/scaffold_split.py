import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles):
  """
  Extracts Bemis-Murcko Scaffolds from SMILES strings and returns them as SMILES
   strings.
  """
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return ""

  try:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)  # Ignores 3D orientation
    return scaffold
  except:
    return ""


def make_scaffold_splits(df, smiles_col='canonical_smiles', test_size=0.2, random_state=42):
  """
  Splits df into train and test sets based on Bemis-Murcko Scaffolds.
  """
  random.seed(random_state)
  df = df.copy()
  df['scaffold'] = df[smiles_col].apply(get_scaffold)
  df = df[df['scaffold'] != ""]
  scaffold_groups = df.groupby('scaffold').groups
  scaffold_keys = list(scaffold_groups.keys())
  random.shuffle(scaffold_keys)  # Shuffle in case of alphabetic or chronological sorting

  train_indices = []
  test_indices = []
  train_target_size = int((1.0 - test_size) * len(df))

  for scaffold in scaffold_keys:
    current_indices = scaffold_groups[scaffold].tolist()
    if len(train_indices) + len(current_indices) <= train_target_size:
      train_indices.extend(current_indices)
    else:
      test_indices.extend(current_indices)

  train_df = df.loc[train_indices].reset_index(drop=True)
  test_df = df.loc[test_indices].reset_index(drop=True)

  return train_df, test_df