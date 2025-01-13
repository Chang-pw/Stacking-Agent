from Stacking_agent.Stacking import *
import argparse
from tqdm import tqdm
from Stacking_agent.tools import *
from Stacking_agent.utils import *
from Stacking_agent.generator import *
import json

def main():
    parser = argparse.ArgumentParser(description="The Stacking framework")
    parser.add_argument('--Task', type=str, help="The chemical task for Agent", required=True)
    parser.add_argument('--tools', type=str, help="The tools for Agent", required=True)
    parser.add_argument('--topN', type=int, help="The top N of the tools for stacking", required=True)
    parser.add_argument('--tool_number', type=int, help="The number of tools for one Agent", required=True)
    parser.add_argument('--train_data_number', type=int, help="The number of training data", required=True)
    parser.add_argument('--no_train',action='store_true', help="You can choose the test agent")

    args = parser.parse_args()

    task = args.Task
    tools = args.tools
    topN = args.topN
    tool_number = args.tool_number
    train_data_number = args.train_data_number
    no_train = args.no_train

    tools =eval(tools)
    score =0
    task_query,task_description = task2query(task)

    Agent_tool.set_description(task_description)
    Agent_tool.set_task_name(task)

    if no_train:
        test_agent = eval(input("请输入列表内容:").strip())
        generator = ToolGenerator()
        final_agent,wo = generator.generate(test_agent)
        if wo:
            result = ['1']
        else:
            result = ['1','2']
    ## Task
    if task == 'Query2SMILES':
        if not no_train:
            with open('./Dataset/Query2SMILES/Query2SMILES_train_data.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()
            final_agent = result[0]['agent_tool']
            print('\n\033[31m ----最终结果---- \033[0m\n')
            print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")
    
        with open('./Dataset/Query2SMILES/Query2SMILES_test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            smiles = i['SMILES']
            description = i['description']
            query = task_query + description 
            if len(result)>1:
                final_answer= final_agent.test_run(query,debug=True)
            else:
                final_answer = final_agent.wo_run(query)
            i['answer'] = final_answer
            blue2 = calculate_BLEU(final_answer,smiles,2)
            print('Final answer:'+ final_answer)
            print('Blue2:'+ str(blue2))
            time.sleep(5)
            score += blue2
        final_score = score/len(test_data)
        print(f"\033[34m 在{task}任务测试集上BLEU-2分数为：'{final_score}'\033[0m")
        
    elif task=='SMILES2Query':
        if not no_train:
            with open(f'./Dataset/SMILES2Query/SMILES2Query_train_data.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()
            final_agent = result[0]['agent_tool']
            print('\n\033[31m ----最终结果---- \033[0m\n')
            print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")
            
        with open('./Dataset/SMILES2Query/SMILES2Query_test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            smiles = i['SMILES']
            description = i['description']
            query = task_query + smiles 
            if len(result)>1:
                final_answer= final_agent.test_run(query,debug=True)
            else:
                final_answer = final_agent.wo_run(query)
            i['answer'] = final_answer
            blue2 = calculate_BLEU(final_answer,description,2)
            print('Final answer:'+ final_answer)
            print('Blue2:'+ str(blue2))
            i['bleu_2'] = blue2
            time.sleep(5)
            score += blue2
        final_score = score/len(test_data)
        print(f"\033[34m 在{task}任务测试集上BLEU-2分数为：'{final_score}'\033[0m")
        

    elif "MolecularPropertyPrediction" in task:
        task_name = task.split('_')[1]
        if not no_train:
            with open(f'./Dataset/MolecularPropertyPrediction/{task_name}/train.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()        
            final_agent = result[0]['agent_tool']
        
        with open(f'./Dataset/MolecularPropertyPrediction/{task_name}/test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            smiles = i['SMILES']
            gold_answer = i['gold_answer']
            query = task_query + smiles 
            if len(result)>1:
                final_answer= final_agent.test_run(query,debug=True)
            else:
                final_answer = final_agent.wo_run(query)
            i['answer'] = final_answer
            if gold_answer in i['answer']:
                i['acc'] = 1
            else:
                i['acc'] = 0
            print('Final answer:'+ final_answer)
            print('Accuracy:'+ str(i['acc']))
            time.sleep(5)
            score += i['acc']
        final_score = score/len(test_data)
        print(f"\033[34m 在{task}任务测试集上BLEU-2分数为：'{final_score}'\033[0m")

    elif task == "ReactionPrediction":
        if not no_train:
            with open(f'./Dataset/ReactionPrediction/ReactionPrediction_train_data.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()
            final_agent = result[0]['agent_tool']
            print('\n\033[31m ----最终结果---- \033[0m\n')
            print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")
            
        with open('./Dataset/ReactionPrediction/ReactionPrediction_test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            smiles = i['SMILES']
            reaction = i['reaction']
            query = task_query + reaction 
            if len(result)>1:
                final_answer= final_agent.test_run(query,debug=True)
            else:
                final_answer = final_agent.wo_run(query)
            i['answer'] = final_answer
            blue2 = calculate_BLEU(final_answer,smiles,2)
            print('Final answer:'+ final_answer)
            print('Blue2:'+ str(blue2))
            i['bleu_2'] = blue2
            time.sleep(5)
            score += blue2
        final_score = score/len(test_data)
        print(f"\033[34m 在{task}任务测试集上BLEU-2分数为：'{final_score}'\033[0m")
    elif task == "YieldPrediction":
        if not no_train:
            with open(f'./Dataset/ReactionPrediction/ReactionPrediction_train_data.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()
            final_agent = result[0]['agent_tool']
            print('\n\033[31m ----最终结果---- \033[0m\n')
            print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")
            
        with open('./Dataset/YieldPrediction/YieldPrediction_test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            gold_answer = i['gold_answer']
            reaction = i['reaction']
            query = task_query + reaction 
            print(query)
            if len(result)>1:
                final_answer= final_agent.test_run(query,debug=True)
            else:
                final_answer = final_agent.wo_run(query)
            i['answer'] = final_answer
            if gold_answer in i['answer']:
                i['acc'] = 1
            else:
                i['acc'] = 0
            print('Final answer:'+ final_answer)
            print('Acc:'+ str(i['acc']))
            time.sleep(5)
            score += i['acc']
        final_score = score/len(test_data)
        print(f"\033[34m 在{task}任务测试集上BLEU-2分数为：'{final_score}'\033[0m")
    try:
        with open(f"./Result/{task}/{result[0]['tool']}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
            json.dump(test_data,f,ensure_ascii=False,indent=4)
    except:
        with open(f"./Result/{task}/{str(test_agent)}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
            json.dump(test_data,f,ensure_ascii=False,indent=4)   
if __name__ == '__main__':
    main()







