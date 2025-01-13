import json
import random


with open('/data1/bowei/agent/my_agent/Dataset/ReactionPrediction copy/ReactionPrediction_train_data.json','r',encoding='utf-8') as f:
    data = json.load(f)

new_data = []
for i in data:
    reaction = i['reaction']
    SMILES = i['SMILES']

    reaction=reaction.replace('___',SMILES)
    
    r = reaction.split('>>')[0]
    segments = r.split('.')
    segments = [segment for segment in segments if segment]
    chosen_index = random.choice(range(len(segments)))
    chosen_s = segments[chosen_index]  
    segments[chosen_index] = '___'  

    reaction = '.'.join(segments) + '>>' + SMILES
    new_data.append({'reaction':reaction,'SMILES':chosen_s})



print(new_data[0])
with open('/data1/bowei/agent/my_agent/Dataset/ReactionPrediction copy/ReactionPrediction_train_data.json','w',encoding='utf-8') as f:
    json.dump(new_data,f,indent=4)
# 
