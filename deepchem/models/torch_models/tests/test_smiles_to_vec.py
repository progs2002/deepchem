import os
import pytest
from deepchem.feat import create_char_to_idx

try:
    import torch
    from deepchem.models.torch_models.smiles_to_vec import Smiles2VecModel
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass

@pytest.mark.torch
def get_char_to_idx(max_len=250):
    dataset_file = os.path.join(os.path.dirname(__file__), "assets",
                                "chembl_25_small.csv")
    return create_char_to_idx(dataset_file, 
                              max_len=max_len, 
                              smiles_field="smiles")

@pytest.mark.torch
def test_smiles_to_vec_base():
    char_to_idx = get_char_to_idx()
    model = Smiles2VecModel(char_to_idx)

    assert model is not None