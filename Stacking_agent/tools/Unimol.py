import lmdb
import numpy as np
import os
import pickle
import pandas as pd
import argparse
from rdkit import Chem

from unimol_tools import MolTrain, MolPredict
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def check_smiles(smi, is_train, smi_strict):
        """
        Validates a SMILES string and decides whether it should be included based on training mode and strictness.

        :param smi: (str) The SMILES string to check.
        :param is_train: (bool) Indicates if this check is happening during training.
        :param smi_strict: (bool) If true, invalid SMILES strings raise an error, otherwise they're logged and skipped.

        :return: (bool) True if the SMILES string is valid, False otherwise.
        :raises ValueError: If the SMILES string is invalid and strict mode is on.
        """
        if Chem.MolFromSmiles(smi) is None:
            if is_train and not smi_strict:
                print(f'Illegal SMILES clean: {smi}')
                return False
            else:
                raise ValueError(f'SMILES rule is illegal: {smi}')
        return True    
def lmdb2csv(lmdb_path):
    target_type = set()
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    result = []
    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        mol = {}
        if Chem.MolFromSmiles(data['smi'])is None:
            print("wrong smiles")
            continue
        mol["SMILES"] = data['smi']
        mol["TARGET"] = data['target'][0]
        if data['target'][0] not in target_type:
            target_type.add(data['target'][0])
        result.append(mol)
    
    df = pd.DataFrame(result)
    df.to_csv(lmdb_path.rsplit('.', 1)[0]+".csv", index=False)
    return lmdb_path.rsplit('.', 1)[0]+".csv",len(target_type)

def get_data(file_path):
    train_csv_path,target_type_length = lmdb2csv(os.path.join(file_path,"train.lmdb"))
    val_csv_path,_= lmdb2csv(os.path.join(file_path,"valid.lmdb"))
    test_csv_path,_ = lmdb2csv(os.path.join(file_path,"test.lmdb"))
    df1 = pd.read_csv(train_csv_path)
    df2 = pd.read_csv(val_csv_path)
    df_merged = pd.concat([df1, df2], ignore_index=True)
    df_merged.to_csv(train_csv_path.rsplit('.', 1)[0]+".csv", index=False)
    if target_type_length > 2:
        target_type = "multilabel_classification"
    else:
        target_type = "classification"
    return train_csv_path, test_csv_path, target_type

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train_unimol")
    parser.add_argument('--file_path', type=str, help="read_path",default="../../Dataset/molecular_property_prediction/hiv")
    args = parser.parse_args()
    file_path = args.file_path
    train_csv_path, test_csv_path, target_type = get_data(file_path)
    print(target_type)
    clf = MolTrain(task= target_type, #multiclass #classification # multilabel_classification
                    data_type='molecule', 
                    epochs=10, 
                    batch_size=16, 
                    metrics='auc',
                    model_name='unimolv2', # avaliable: unimolv1, unimolv2
                    model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                    )
    pred = clf.fit(data = train_csv_path )
    clf = MolPredict(load_model='../exp')
    res = clf.predict(data = test_csv_path)