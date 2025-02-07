from Stacking_agent.Stacking import *
import argparse
from tqdm import tqdm
from Stacking_agent.tools import *
from Stacking_agent.utils import *
from Stacking_agent.generator import *
import json
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import time

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
    debug = True
    tools =eval(tools)
    score =0
    task_query,task_description = task2query(task)
    Agent_tool.set_description(task_description)
    Agent_tool.set_task_name(task)
    generator = ToolGenerator()
    
    if no_train:
        test_agent = input("请输入列表内容:").strip()
        final_agent,wo = generator.generate(test_agent)

    ## Task
    if task == 'Molecule_Design':
        if not no_train:
            with open('./Dataset/Molecule_Design/train.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()
            final_agent,wo = generator.generate(result[0]['tool'])
            print('\n\033[31m ----最终结果---- \033[0m\n')
            print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")
    
        with open('./Dataset/Molecule_Design/test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        # with open("./Result/Stacking/Query2SMILES/[['ChemDFM_0', 'Name2SMILES_1'], 'ChemDFM_0']_5_2_10.json",'r',encoding='utf-8')    as f:
            # test_data = json.load(f)
        for i in tqdm(test_data):
            start_time = time.time()
            smiles = i['SMILES']
            description = i['description']
            query = task_query + description 
            if not wo:
                final_answer,all_tokens= final_agent.test_run(query,debug=debug)
            else:
                final_answer,all_tokens = final_agent.wo_run(query)

            end_time = time.time()
            
            i['answer'] = final_answer
            i['all_tokens'] = all_tokens
            i['time']=round(end_time - start_time, 3)
            blue2 = calculate_BLEU(final_answer,smiles,2)
            print('Final answer:'+ final_answer)
            print('Blue2:'+ str(blue2))
            i['exact'] = calculate_exact(final_answer,smiles)
            i['blue-2'] = blue2
            i['Dis'] = calculate_dis(final_answer,smiles)
            i['Validity'] = calculate_Validity(final_answer)
            i['MACCS'],i['RDK'],i['Morgan']=calculate_FTS(final_answer,smiles)
            time.sleep(5)
            score += blue2
            try:
                with open(f"./Result/Stacking/{task}/{result[0]['tool']}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)
            except:
                with open(f"./Result/Stacking/{task}/{str(test_agent)}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)   
        final_score = score/len(test_data)
        print(f"Exact: {calculate_avg(test_data,'exact')}")
        print(f"Blue-2: {final_score}")
        print(f"Dis: {calculate_avg(test_data,'Dis')}")
        print(f"Validity: {calculate_avg(test_data,'Validity')}")
        print(f"MACCS: {calculate_avg(test_data,'MACCS')}")
        print(f"RDK: {calculate_avg(test_data,'RDK')}")
        print(f"Morgan: {calculate_avg(test_data,'Morgan')}")
    elif task=='Molecule_captioning':
        if not no_train:
            with open(f'./Dataset/Molecule_captioning/train.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()
            final_agent,wo = generator.generate(result[0]['tool'])
            print('\n\033[31m ----最终结果---- \033[0m\n')
            print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")

        with open('./Dataset/Molecule_captioning/test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            start_time = time.time()
            smiles = i['SMILES']
            description = i['description']
            query = task_query + smiles 
            if not wo:
                final_answer,all_tokens= final_agent.test_run(query,debug=debug)
            else:
                final_answer,all_tokens = final_agent.wo_run(query)

            end_time = time.time()
            i['answer'] = final_answer
            i['all_tokens'] = all_tokens
            i['time']=round(end_time - start_time, 3)
            blue2 = calculate_BLEU(final_answer,description,2)
            print('Final answer:'+ final_answer)
            print('Blue2:'+ str(blue2))
            i['bleu_2'] = blue2
            i['bleu_4'] = calculate_BLEU(final_answer,description,4)
            i['rouge_2'],i['rouge_4'],i['rouge_L'] = calculate_rouge(final_answer,description)
            i['meteor'] = calculate_meteor(final_answer,description)
            time.sleep(5)
            score += blue2
            try:
                with open(f"./Result/Stacking/{task}/{result[0]['tool']}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)
            except:
                with open(f"./Result/Stacking/{task}/{str(test_agent)}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)   
        final_score = score/len(test_data)
        print(f"Bleu_2: {final_score}")
        print(f"Bleu_4: {calculate_avg(test_data,'bleu_4')}")
        print(f"rouge_2: {calculate_avg(test_data,'rouge_2')}")
        print(f"rouge_4: {calculate_avg(test_data,'rouge_4')}")
        print(f"rouge_L: {calculate_avg(test_data,'rouge_L')}")
        print(f"meteor: {calculate_avg(test_data,'meteor')}")

    elif "MolecularPropertyPrediction" in task:
        task_name = task.split('_')[1]
        if not no_train:
            with open(f'./Dataset/MolecularPropertyPrediction/{task_name}/train.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()        
            final_agent,wo = generator.generate(result[0]['tool'])
        
        with open(f'./Dataset/MolecularPropertyPrediction/{task_name}/test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            start_time = time.time()
            smiles = i['SMILES']
            gold_answer = i['gold_answer']
            query = task_query + smiles 
            if not wo:
                final_answer,all_tokens= final_agent.test_run(query,debug=debug)
            else:
                final_answer,all_tokens = final_agent.wo_run(query)

            end_time = time.time()
            i['answer'] = final_answer
            i['all_tokens'] = all_tokens
            i['time']=round(end_time - start_time, 3)
            print('Final answer:'+ final_answer)
            time.sleep(5)
            try:
                with open(f"./Result/Stacking/{task}/{result[0]['tool']}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)
            except:
                with open(f"./Result/Stacking/{task}/{str(test_agent)}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)   
        y_true = [1 if i['gold_answer']=='Yes' else 0 for i in test_data]
        y_pred = [1 if i['answer']=='Yes' else 0 for i in test_data]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        final_score = auc(fpr, tpr)
        # final_score = f1_score(y_true, y_pred,zero_division=1.0)
        print(f"\033[34m 在{task}任务测试集上AUC-ROC分数为：'{final_score}'\033[0m")

    elif task == "ReactionPrediction":
        if not no_train:
            with open(f'./Dataset/ReactionPrediction/train.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()
            final_agent,wo = generator.generate(result[0]['tool'])
            print('\n\033[31m ----最终结果---- \033[0m\n')
            print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")
            
        with open('./Dataset/ReactionPrediction/test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            start_time = time.time()
            smiles = i['SMILES']
            reaction = i['reaction']
            query = task_query + reaction 
            if not wo:
                final_answer,all_tokens= final_agent.test_run(query,debug=debug)
            else:
                final_answer,all_tokens = final_agent.wo_run(query)

            end_time = time.time()
            i['answer'] = final_answer            
            i['all_tokens'] = all_tokens
            i['time']=round(end_time - start_time, 3)
            blue2 = calculate_BLEU(final_answer,smiles,2)
            print('Final answer:'+ final_answer)
            print('Blue2:'+ str(blue2))
            i['bleu_2'] = blue2
            time.sleep(5)
            score += blue2
            try:
                with open(f"./Result/Stacking/{task}/{result[0]['tool']}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)
            except:
                with open(f"./Result/Stacking/{task}/{str(test_agent)}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)   
        final_score = score/len(test_data)
        print(f"\033[34m 在{task}任务测试集上Acc分数为：'{final_score}'\033[0m")
        
    elif "YieldPrediction" in task:
        task_name = task.split('_')[1]
    
        if not no_train:
            with open(f'./Dataset/YieldPrediction/{task_name}/train.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()
            final_agent,wo = generator.generate(result[0]['tool'])
            print('\n\033[31m ----最终结果---- \033[0m\n')
            print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")
            
        with open(f'./Dataset/YieldPrediction/{task_name}/test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            start_time = time.time()
            gold_answer = i['gold_answer']
            reaction = i['Reaction']
            query = task_query + reaction 
            if not wo:
                final_answer,all_tokens= final_agent.test_run(query,debug=debug)
            else:
                final_answer,all_tokens = final_agent.wo_run(query)

            end_time = time.time()
            i['answer'] = final_answer
            i['all_tokens'] = all_tokens
            i['time']=round(round(end_time - start_time, 3), 3)
            if gold_answer in i['answer']:
                i['acc'] = 1
            else:
                i['acc'] = 0
            print('Final answer:'+ final_answer)
            print('Acc:'+ str(i['acc']))
            
            time.sleep(5)
            score += i['acc']
            try:
                with open(f"./Result/Stacking/{task}/{result[0]['tool']}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)
            except:
                with open(f"./Result/Stacking/{task}/{str(test_agent)}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)   
        final_score = score/len(test_data)
        print(f"\033[34m 在{task}任务测试集上Acc分数为：'{final_score}'\033[0m")

    elif task == "Retrosynthesis":
        if not no_train:
            with open(f'./Dataset/Retrosynthesis/train.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()
            final_agent,wo = generator.generate(result[0]['tool'])
            print('\n\033[31m ----最终结果---- \033[0m\n')
            print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")
            
        with open('./Dataset/Retrosynthesis/test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            start_time = time.time()
            product = i['input']
            gold_answer = i['gold_answer']
            query = task_query + product 

            if not wo:
                final_answer,all_tokens= final_agent.test_run(query,debug=debug)
            else:
                final_answer,all_tokens = final_agent.wo_run(query)

            end_time = time.time()
            i['all_tokens'] = all_tokens
            i['time']=round(end_time - start_time, 3)
            i['answer'] = final_answer
            try:
                final_answer_list = eval(final_answer)
                if set(gold_answer) == set(final_answer_list):
                    i['acc'] = 1
                else:
                    i['acc'] = 0
            except:
                i['acc']=0

            print('Final answer:'+ final_answer)
            print('Acc:'+ str(i['acc']))
            time.sleep(5)
            score += i['acc']
            try:
                with open(f"./Result/Stacking/{task}/{result[0]['tool']}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)
            except:
                with open(f"./Result/Stacking/{task}/{str(test_agent)}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)   
        final_score = score/len(test_data)
        print(f"\033[34m 在{task}任务测试集上Acc分数为：'{final_score}'\033[0m")

    elif "ReagentSelection" in task:
        task_name = task.split('_')[1]
    
        if not no_train:
            with open(f'./Dataset/ReagentSelection/{task_name}/train.json','r',encoding='utf-8')    as f:
                train_data = json.load(f)
            result,_ = Stacking(tools=tools, top_n=topN, tool_number=tool_number,train_data=train_data,
                                train_data_number=train_data_number,task=task,query=task_query)._run()
            final_agent,wo = generator.generate(result[0]['tool'])
            print('\n\033[31m ----最终结果---- \033[0m\n')
            print(f"\033[34m 在训练集上表现最优的是 {result[0]['tool']} , 其分数为 {result[0]['score']} 。接下来开始在{task}任务测试集上运行 \033[0m")
            
        with open(f'./Dataset/ReagentSelection/{task_name}/test.json','r',encoding='utf-8')    as f:
            test_data = json.load(f)
        for i in tqdm(test_data):
            start_time = time.time()
            gold_answer = i['gold_answer']
            reaction = i['Reaction']
            choices = i['choices']
            query = task_query.format(reaction=reaction, choices=choices)
            if not wo:
                final_answer,all_tokens= final_agent.test_run(query,debug=debug)
            else:
                final_answer,all_tokens = final_agent.wo_run(query)

            end_time = time.time()

            i['answer'] = final_answer
            i['all_tokens'] = all_tokens
            i['time']=round(end_time - start_time, 3)
            if gold_answer in i['answer']:
                i['acc'] = 1
            else:
                i['acc'] = 0
            print('Final answer:'+ final_answer)
            print('Acc:'+ str(i['acc']))
            
            time.sleep(5)
            score += i['acc']
            try:
                with open(f"./Result/Stacking/{task}/{result[0]['tool']}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)
            except:
                with open(f"./Result/Stacking/{task}/{str(test_agent)}_{topN}_{tool_number}_{train_data_number}.json",'w',encoding='utf-8') as f:
                    json.dump(test_data,f,ensure_ascii=False,indent=4)   
        final_score = score/len(test_data)
        print(f"\033[34m 在{task}任务测试集上Acc分数为：'{final_score}'\033[0m")


    try:
        now = datetime.now()
        text_content = str(result)
        with open(f'./log/{task}_{now}.txt', 'w', encoding='utf-8') as file:
            file.write(text_content)
    except:
        with open(f'./log/{task}_{now}.txt', 'w', encoding='utf-8') as file:
            file.write(str(final_score))

if __name__ == '__main__':
    main()







