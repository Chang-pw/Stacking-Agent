import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

def cosine_similarity(embeddings_1,embeddings_2):
    return embeddings_1 @ embeddings_2.T

## Molecule Captioning: "Description2SMILES" (Bge-M3 description)
def Description2SMILES_train_data():
    with open("./Dataset/molecule_captioning_all.json" ,"r",encoding="utf-8") as f:
        all_data = json.load(f)
    with open("./Dataset/molecule_captioning_test.json","r",encoding="utf-8") as f:
        test_data = json.load(f)
    test_smiles = [item['description'] for item in test_data]

    all_smiles = [item['description'] for item in all_data if item['description'] not in test_smiles]

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    all_embeddings = model.encode(all_smiles, max_length=8192)['dense_vecs']

    test_embeddings = model.encode(test_smiles, max_length=8192)['dense_vecs']


    similarity_matrix = test_embeddings @ all_embeddings.T  # Shape: (100, N), where N is the number of sentences in the "all" dataset

    most_similar_indices = np.argmax(similarity_matrix, axis=1)

    most_similar_sentences = [all_smiles[idx] for idx in most_similar_indices]
    results = []

    for z in all_data:
        if z['description'] in most_similar_sentences:
            results.append(z)

    with open("./Dataset/Description2SMILES_train_data.json","w",encoding="utf-8") as f:
        json.dump(results,f,indent=4)
    print(len(results))



## Molecule Captioning: "SMILES2Description" (Morgan Fingerprint)
def SMILES2Description_train_data():
    with open("./Dataset/molecule_captioning_all.json","r",encoding="utf-8") as f:
        all_data = json.load(f)
    with open("./Dataset/molecule_captioning_test.json","r",encoding="utf-8") as f:
        test_data = json.load(f)
    test_smiles = [item['SMILES'] for item in test_data]
    all_smiles = [item['SMILES'] for item in all_data if item['SMILES'] not in test_smiles]

    mol_test_list = [Chem.MolFromSmiles(smiles) for smiles in test_smiles]
    mol_all_list = [Chem.MolFromSmiles(smiles) for smiles in all_smiles]

    fingerprints_test = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mol_test_list]
    fingerprints_all = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mol_all_list]

    results = []

    for i, test_fp in enumerate(fingerprints_test):
        max_similarity = 0
        most_similar_smiles = ""
        for j, all_fp in enumerate(fingerprints_all):
            similarity = TanimotoSimilarity(test_fp, all_fp)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_smiles = all_smiles[j]

        for z in all_data:
            if z['SMILES'] == most_similar_smiles:
                results.append(z)
                break

    with open("./Dataset/SMILES2Description_train_data.json","w",encoding="utf-8") as f:
        json.dump(results,f,indent=4)
    print(len(results))


if __name__ == "__main__":
    Description2SMILES_train_data()
    SMILES2Description_train_data()

    pass
