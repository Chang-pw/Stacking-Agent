import json
import numpy as np
import csv


with open('/data1/bowei/agent/my_agent/Result/Stacking/ReagentSelection_ligand/GPT4o_0shot.json','r',encoding='utf-8') as f:
    test_data = json.load(f)

with open('/data1/bowei/agent/my_agent/Dataset/ReagentSelection/ligand/test.json','r',encoding='utf-8') as f:
    ground_truth = json.load(f)



for index,i in enumerate(test_data):
    i['gold_answer'] = ground_truth[index]['gold_answer']
    if i['gold_answer'] in i ['answer']:
        i['acc'] = 1
    else:
        i['acc'] = 0

with open('/data1/bowei/agent/my_agent/Result/Stacking/ReagentSelection_ligand/GPT4o_0shot.json','w',encoding='utf-8') as f:
    json.dump(test_data,f,indent=2)
