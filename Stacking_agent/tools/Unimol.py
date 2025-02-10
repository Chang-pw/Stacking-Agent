import lmdb
import numpy as np
import os
import pickle
import pandas as pd
import argparse
from rdkit import Chem
import json

from unimol_tools import MolTrain, MolPredict
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  
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

def get_data_json(file_path):
    label_dict = {}
    label_dict["Yes"] = 1
    label_dict["No"] = 0
    with open(os.path.join(file_path,"test.json"),"r") as fp:
        data = json.load(fp)
    result = []
    for i in data:
        mol = {}
        if Chem.MolFromSmiles(i['SMILES'])is None:
            print("wrong smiles")
            continue
        mol["SMILES"] = i['SMILES']
        mol["TARGET"] = label_dict[i['gold_answer']]
        result.append(mol)
    df = pd.DataFrame(result)
    df.to_csv(os.path.join(file_path,"test.csv"), index=False)
    return os.path.join(file_path,"test.csv")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train_unimol")
    parser.add_argument('--file_path', type=str, help="read_path",default="../../Dataset/MolecularPropertyPrediction/bbbp")
    parser.add_argument('--task', type=str, help="train or test",default="test")
    parser.add_argument('--model_path', type=str, help="where your finetuned model")
    args = parser.parse_args()
    file_path = args.file_path
    task = args.task
    if task == "train":
        train_csv_path, test_csv_path, target_type = get_data(file_path)
        clf = MolTrain( task= target_type, #multiclass #classification # multilabel_classification
                        data_type='molecule', 
                        epochs=10, 
                        batch_size=16, 
                        metrics='auc',
                        model_name='unimolv2', # avaliable: unimolv1, unimolv2
                        model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                        save_path = f'./exp_{file_path.split("/")[-1]}'
                        )
        pred = clf.fit(data = train_csv_path )
        clf = MolPredict(load_model=f'./exp_{file_path.split("/")[-1]}')
        res = clf.predict(data = test_csv_path)
    elif task == "test":
        test_csv_path = get_data_json( file_path )
        clf = MolPredict(load_model=args.model_path)
        res = clf.predict(data = test_csv_path)