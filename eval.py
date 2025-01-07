from Stacking_agent.Stacking import *
from tqdm import tqdm
from Stacking_agent.tools import *
from Stacking_agent.utils import *
import json

def eval(task='',tool_list=[]):
    task_query = task2query(task)
    if task == 'Query2SMILES':
        final_agent = Agent(tool_list)
        name = str(final_agent)
        score =0
        with open('./Dataset/molecule_captioning_test.json','r',encoding='utf-8')  as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            smiles = i['SMILES']
            description = i['description']
            query = task_query + description 
            final_agent.buffer = False
            final_agent.debug = True
            final_answer,r,h = final_agent._run(query)
            i['answer'] = final_answer
            blue2 = calculate_BLEU(final_answer,smiles,2)
            print('Final answer:'+ final_answer)
            print('Blue2:'+ str(blue2))
            time.sleep(5)
            score += blue2
    
            with open(f'./Result/eval/{task}_{tool_list[0].name}.json','w',encoding='utf-8') as f:
                json.dump(test_data,f,indent=4,ensure_ascii=False)
        final_score = score/len(test_data)
        print(f"\033[34m 在{task}任务测试集上BLEU-2分数为：'{final_score}'\033[0m")

if __name__ == '__main__':
    task = 'Query2SMILES'
    eval(task,[ChemDFM()])
    

    # with open('./Result/eval/Query2SMILES_Name2SMILES tool.json','r',encoding='utf-8') as f:
    #     test_data = json.load(f)
    # score =0 
    # for i in test_data:
    #     if '\u001b[0m' in i['answer']:
    #         i['answer'] = i['answer'].split('\u001b[0m')[-1]
    #     i['Blue2'] = calculate_BLEU(i['answer'],i['SMILES'],2)
    #     score+=i['Blue2']

    # print(score/100)
    

    
