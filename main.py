from Stacking_agent.Stacking import *
import argparse
from tqdm import tqdm
from Stacking_agent.tools import *
from Stacking_agent.utils import *
import json

def main():
    parser = argparse.ArgumentParser(description="The Stacking framework")
    parser.add_argument('--Task', type=str, help="The chemical task for Agent", required=True)
    parser.add_argument('--tools', type=str, help="The tools for Agent", required=True)
    parser.add_argument('--topN', type=int, help="The top N of the tools for stacking", required=True)
    parser.add_argument('--tool_number', type=int, help="The number of tools for one Agent", required=True)
    parser.add_argument('--train_data_number', type=int, help="The number of training data", required=True)

    args = parser.parse_args()

    task = args.Task
    tools = args.tools
    topN = args.topN
    tool_number = args.tool_number
    train_data_number = args.train_data_number

    tools =eval(tools)
    score =0
    task_query = task2query(task)
    
    ## Task
    if task == 'Query2SMILES':
        with open('./Dataset/Description2SMILES_train_data.json','r',encoding='utf-8')    as f:
            train_data = json.load(f)
        result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                            train_data_number=train_data_number,query=task_query)._run()
        final_agent = result[0]['agent_tool']
        try:
            final_agent.set_all_buffers()
        except:
            pass
        print('\n\033[31m ----最终结果---- \033[0m\n')
        print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")
    
        with open('./Dataset/molecule_captioning_test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            smiles = i['SMILES']
            description = i['description']
            query = task_query + description 
            final_agent.debug = True
            if len(result)>1:
                final_answer = final_agent._run(query)
            else:
                final_answer = final_agent.wo_run(query)
            i['answer'] = final_answer
            blue2 = calculate_BLEU(final_answer,smiles,2)
            print('Final answer:'+ final_answer)
            print('Blue2:'+ str(blue2))
            time.sleep(5)
            score += blue2
        final_score = score/len(test_data)
        print(f"\033[34m {result[0]['tool']}在{task}任务测试集上BLEU-2分数为：'{final_score}'\033[0m")
        
    elif task=='SMILES2Query':
        with open('./Dataset/SMILES2Description_train_data.json','r',encoding='utf-8')    as f:
            train_data = json.load(f)
        result,_ = Stacking(task=task, tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data)._run()
        final_agent = result[0]['agent_tool']
        print('\n\033[31m ----最终结果---- \033[0m\n')
        print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")

        with open('./Dataset/molecule_captioning_test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
            
        for i in tqdm(test_data):
            smiles = i['SMILES']
            description = i['description']
            query = 'Please show me a description of this molecule:' + smiles
            final_agent.debug = True
            final_answer, response, history = final_agent._run(query,[],debug=False)
            i['answer'] = final_answer
            blue2 = calculate_BLEU(final_answer,description,2)
            print('Final answer:'+ final_answer)
            print('Blue2:'+ str(blue2))
            time.sleep(15)
            score += blue2
        final_score = score/len(test_data)
        print(f"\033[34m{result[0]['tool']}在{task}任务测试集上BLEU-2分数为：'{final_score}'\033[0m")


    with open(f"./Result/{task}_{result[0]['tool']}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
        json.dump(test_data,f,ensure_ascii=False,indent=4)
    
if __name__ == '__main__':
    main()







