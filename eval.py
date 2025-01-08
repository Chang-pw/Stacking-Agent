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
        name = str(final_agent)
        score =0
        with open('./Dataset/molecule_captioning_test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            smiles = i['SMILES']
            description = i['description']
            query = task_query + description 
            final_agent.debug = True
            if len(agent_list)>1:
                final_answer = final_agent.test_run(query)
            else:
                final_answer = final_agent.wo_run(query)
            i['answer'] = final_answer
            blue2 = calculate_BLEU(final_answer,smiles,2)
            print('Final answer:'+ final_answer)
            print('Blue2:'+ str(blue2))
            time.sleep(5)
            score += blue2
        final_score = score/len(test_data)
        # print(f"\033[34m {agent_list[0]['tool']}在{task}任务测试集上BLEU-2分数为：'{final_score}'\033[0m")

if __name__ == '__main__':
    task = 'Query2SMILES'
    task_query,task_description = task2query(task)
    Agent_tool.set_description(task_description)
    Agent_tool.set_task_name(task)
    tool = [Name2SMILES()]
    agent1 = Agent(tool)
    agent_1 = Agent_tool(agent=agent1)
    agent2 = Agent([agent_1])
    agent_2 = Agent_tool(agent=agent2).test_run('What is the SMILES of Bromocriptine')
    
    # eval(task,agent_list=[agent_2,agent_1])    

    
