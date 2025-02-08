import lmdb
import numpy as np
import os
import pickle
import pandas as pd
import argparse

from unimol_tools import MolTrain, MolPredict

def lmdb2csv(lmdb_path):
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
        mol["SMILES"] = data['smi']
        mol["TARGET"] = data['target'][0]
        result.append(mol)
    
    df = pd.DataFrame(result)
    df.to_csv(lmdb_path.rsplit('.', 1)[0]+".csv", index=False)
    return lmdb_path.rsplit('.', 1)[0]+".csv"

def get_data(file_path):
    train_csv_path = lmdb2csv(os.path.join(file_path,"train.lmdb"))
    val_csv_path = lmdb2csv(os.path.join(file_path,"valid.lmdb"))
    test_csv_path = lmdb2csv(os.path.join(file_path,"test.lmdb"))
    df1 = pd.read_csv(train_csv_path)
    df2 = pd.read_csv(val_csv_path)
    df_merged = pd.concat([df1, df2], ignore_index=True)
    df_merged.to_csv(train_csv_path.rsplit('.', 1)[0]+".csv", index=False)
    return train_csv_path,test_csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train_unimol")
    parser.add_argument('--file_path', type=str, help="read_path",default="../../Dataset/molecular_property_prediction/bbbp")
    args = parser.parse_args()
    file_path = args.file_path
    train_csv_path,test_csv_path = get_data(file_path)
    clf = MolTrain(task='classification', 
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