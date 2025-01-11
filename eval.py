from Stacking_agent.Stacking import *
from tqdm import tqdm
from Stacking_agent.tools import *
from Stacking_agent.utils import *
import json

def eval(task='',agent_list=[]):
    task_query,task_description = task2query(task)
    Agent_tool.set_description(task_description)
    Agent_tool.set_task_name(task)
    if task == 'Query2SMILES':
        final_agent = agent_list[0]
        score =0
        with open('./Dataset/molecule_captioning_test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data[85:]):
            smiles = i['SMILES']
            description = i['description']
            query = task_query + description 
            final_agent.debug = True

            final_answer = final_agent.test_run(query,debug=True)

            i['answer'] = final_answer
            blue2 = calculate_BLEU(final_answer,smiles,2)
            print('Final answer:'+ final_answer)
            print('Blue2:'+ str(blue2))
            time.sleep(5)
            score += blue2
            with open('./Final result/molecule_captioning_test.json','w',encoding='utf-8') as f:
                json.dump(test_data,f,indent=4)
        final_score = score/len(test_data)
        print(f"BLEU-2分数为：'{final_score}'")

if __name__ == '__main__':
    task = 'SMILES2Query'
    task_query,task_description = task2query(task)
    Agent_tool.set_description(task_description)
    Agent_tool.set_task_name(task)
    n2s_0 = Name2SMILES()
    n2s_1 = Agent_tool(Agent([n2s_0]))
    n2s_2 = Agent_tool(Agent([n2s_1,n2s_0]))
    n2s_3 = Agent_tool(Agent([n2s_2,n2s_0]))
    dfm_0 = ChemDFM()
    Final_agent = Agent_tool(Agent([n2s_3,dfm_0]))
    eval(task,agent_list=[Final_agent])    

    
