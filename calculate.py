from Stacking_agent.utils import *
import json
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc


def Query2SMILES(data):
    with open(f"./Result/Stacking/Molecule_Design/{data}.json",'r',encoding='utf-8') as f:
        test_data = json.load(f)

    for i in test_data:
        smiles = i['SMILES']
        final_answer = i['answer']
        if final_answer == None:
            final_answer = ''

        i['exact'] = calculate_exact(final_answer,smiles)
        i['blue-2'] = calculate_BLEU(final_answer,smiles,2)
        i['Dis'] = calculate_dis(final_answer,smiles)
        i['Validity'] = calculate_Validity(final_answer)
        i['MACCS'],i['RDK'],i['Morgan']=calculate_FTS(final_answer,smiles)

    with open(f"./Result/Stacking/Molecule_Design/{data}.json",'w',encoding='utf-8') as f:
        json.dump(test_data,f,indent=2)
        
    print(f"Exact: {calculate_avg(test_data,'exact')}")
    print(f"Blue-2: {calculate_avg(test_data,'blue-2')}")
    print(f"Dis: {calculate_avg(test_data,'Dis')}")
    print(f"Validity: {calculate_avg(test_data,'Validity')}")
    print(f"MACCS: {calculate_avg(test_data,'MACCS')}")
    print(f"RDK: {calculate_avg(test_data,'RDK')}")
    print(f"Morgan: {calculate_avg(test_data,'Morgan')}")
    print(f"All: {calculate_avg(test_data,'all_tokens')}")
    print(f"time: {calculate_avg(test_data,'time')}")

def Molecule_captioning(data):
    with open(f"./Result/Stacking/Molecule_captioning/{data}.json",'r',encoding='utf-8') as f:
        test_data = json.load(f)

    for i in test_data:
        description = i['description']
        final_answer = i['answer']
        if final_answer == None:
            final_answer = ''
        i['bleu_2'] = calculate_BLEU(final_answer,description,2)
        i['bleu_4'] = calculate_BLEU(final_answer,description,4)
        i['rouge_2'],i['rouge_4'],i['rouge_L'] = calculate_rouge(final_answer,description)
        i['meteor'] = calculate_meteor(final_answer,description)

    with open(f"./Result/Stacking/Molecule_captioning/{data}.json",'w',encoding='utf-8') as f:
        json.dump(test_data,f,indent=2)
        
    print(f"Bleu_2: {calculate_avg(test_data,'bleu_2')}")
    print(f"Bleu_4: {calculate_avg(test_data,'bleu_4')}")
    print(f"rouge_2: {calculate_avg(test_data,'rouge_2')}")
    print(f"rouge_4: {calculate_avg(test_data,'rouge_4')}")
    print(f"rouge_L: {calculate_avg(test_data,'rouge_L')}")
    print(f"meteor: {calculate_avg(test_data,'meteor')}")


def MolecularPropertyPrediction(data,task):
    with open(f"./Result/Stacking/MolecularPropertyPrediction_{task}/{data}.json",'r',encoding='utf-8') as f:
        test_data = json.load(f)
    y_true = [1 if i['gold_answer']=='Yes' else 0 for i in test_data]
    y_pred = [1 if i['answer']=='Yes' else 0 for i in test_data]

    f1 = f1_score(y_true, y_pred,zero_division=1.0)
    print(f"F1 score: {f1}")
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

def YieldPrediction(data,task):
    from sklearn.metrics import accuracy_score

    with open(f"./Result/YieldPrediction_{task}/{data}.json",'r',encoding='utf-8') as f:
        test_data = json.load(f)
    score=0
    # for i in test_data:
        # score += i['acc']
    y_true = [1 if i['gold_answer']=='Yes' else 0 for i in test_data]
    y_pred = [1 if i['answer']=='Yes' else 0 for i in test_data]
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    # print(f"Accuracy: {score/len(test_data)}")

def ReagentSelection(data,task):
    from sklearn.metrics import accuracy_score

    with open(f"./Result/Stacking/ReagentSelection_{task}/{data}.json",'r',encoding='utf-8') as f:
        test_data = json.load(f)
    score=0
    for i in test_data:
        score += i['acc']
    accuracy=score/100
    # accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    # print(f"Accuracy: {score/len(test_data)}")

def ReactionPrediction(data):
    with open(f"./Result/Stacking/ReactionPrediction/{data}.json",'r',encoding='utf-8') as f:
        test_data = json.load(f)
    score = 0
    for i in test_data:
        if i['SMILES'] in i['answer']:
            score+=1
    print(f"Accuracy: {score/len(test_data)}")

def MultiAgent(data,avg=False):
    if avg:
        list_ = ['Chain','Random','FullConnected','Star','Layered','Debate']
        tokens = 0
        time = 0
        for i in list_:
            with open(f"./Result/Multiagent/Tool/{i}_8_None.json",'r',encoding='utf-8') as f:
                test_data = json.load(f)
            all_time = 0
            all_tokens = 0
            for i in test_data:
                all_time+=i['time']
                all_tokens+=i['all_tokens']
            
            tokens+=all_tokens/len(test_data)
            time+=all_time/len(test_data)

        print(f"tokens: {tokens/len(list_)}")
        print(f"time: {time/len(list_)}")
        return None


    with open(f"./Result/Multiagent/Tool/{data}.json",'r',encoding='utf-8') as f:
        test_data = json.load(f)
    all_time = 0
    all_tokens = 0
    for i in test_data:
        # print(i)
        final_answer = i['answer'].strip("\"")
        smiles = i['SMILES']
        i['exact'] = calculate_exact(final_answer,smiles)
        i['blue-2'] = calculate_BLEU(final_answer,smiles,2)
        i['Dis'] = calculate_dis(final_answer,smiles)
        i['Validity'] = calculate_Validity(final_answer)
        i['MACCS'],i['RDK'],i['Morgan']=calculate_FTS(final_answer,smiles)
        all_time+=i['time']
        all_tokens+=i['all_tokens']


    print(f"Exact: {calculate_avg(test_data,'exact')}")
    print(f"Blue-2: {calculate_avg(test_data,'blue-2')}")
    print(f"Dis: {calculate_avg(test_data,'Dis')}")
    print(f"Validity: {calculate_avg(test_data,'Validity')}")
    print(f"MACCS: {calculate_avg(test_data,'MACCS')}")
    print(f"RDK: {calculate_avg(test_data,'RDK')}")
    print(f"Morgan: {calculate_avg(test_data,'Morgan')}")
    print(f"time: {all_time/len(test_data)}")
    print(f"all_tokens: {all_tokens/len(test_data)}")
    print(len(test_data))

if __name__ == '__main__':
    # Query2SMILES("[['ChemDFM_1','Name2SMILES_1'],['ChemDFM_1','Name2SMILES_2']]_5_2_10")
    # Molecule_captioning("SMILES2Description_2_5_2_10")
    # MolecularPropertyPrediction("ChemDFM_1_5_2_10",'tox21')
    # ReagentSelection("GPT4o_0shot","solvent")
    # ReactionPrediction("[['ChemDFM_0', 'SMILES2Property_1'], ['ChemDFM_1', 'ChemDFM_0']]_5_2_30")
    # MolecularPropertyPrediction("[['SMILES2Property_1','ChemDFM_0'],'ChemDFM_1']_5_2_10",'tox21')
    MultiAgent('Random_8_None',avg=False)
