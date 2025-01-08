from .warmup import *
from .agent import Agent
from .utils import calculate_BLEU,sorted_tools
from .tools import *
import random
import json
import time
from tqdm import tqdm
import itertools
import dill
import math
import re

class Stacking:
    def __init__(self,tools:list,top_n:int,tool_number=2,train_data=[],train_data_number=10,query=""):
        self.all_tools = tools
        self.tool_number = tool_number
        self.query = query
        self.Warmup = Warmup(self.all_tools,tool_number=self.tool_number,data=train_data,train_data_number=train_data_number,query=self.query)
        self.data = self.Warmup.sample_data
        self.warmup = sorted_tools(self.Warmup._run())
        self.top_n = top_n


    def test(self,tool_list,name1,name2):
        test_agent = Agent(tool_list)
        test_data = self.data
        score =0 
        for i in tqdm(test_data):
            smiles = i['SMILES']
            description = i['description']
            query = self.query + description
            final_answer, response, history = test_agent._run(query,[],debug=False)
            i['answer'] = final_answer
            i['blue2'] = calculate_BLEU(final_answer,smiles,2)
            time.sleep(5)
            score += i['blue2']
        blue2 = score/len(test_data)
        return test_agent,blue2,test_data


    def one_Stacking(self,tool_list):
        """ Note that Input the Sorted tool list """
        # Top N tools: tools
        tools = tool_list[:self.top_n]
        # Best performance tool: tool_1
        tool_1 = tools[0]
        # Other tools for stacking : tools 
        remaining_tools = tools[1:]
        # Result list
        result_list = []

        tool_number = self.tool_number
        if tool_number > len(tools):
            print(f"由于tool_number设置大于topN的工具列表，将tool_number设置为最大值 {len(tools)}.\n")
            tool_number = len(tools)
        try:
            for i in remaining_tools:
                match = re.match(r'^(.*)_\d+$', tool_1['tool'])
                if i['tool'] == f'{match.group(1)}_0':
                    remaining_tools.remove(i)
                    print(f"由于首选工具{tool_1['tool']}在预热阶段已经与工具{match.group(1)}_0叠加过，则将排除该基础工具 \n")
        except:
            pass
        
        tool_combinations = itertools.combinations(remaining_tools, tool_number - 1)
        combination_number = math.comb(len(remaining_tools), tool_number - 1)

        print(f"首选工具为{tool_1['tool']}, 剩余top{self.top_n}的工具为{[i['tool'] for i in remaining_tools]}\n")

        print(f"将从剩余的工具中选择 {tool_number - 1} 个工具，与 {tool_1['tool']} 组合，共生成 {combination_number} 种组合。\n")

        for combination in tool_combinations:
            # Combine tool_1 with the remaininged cosmbination of tools
            tool_combination = [tool_1['agent_tool']] + [tool['agent_tool'] for tool in combination]
            
            # Get the names of the tools in the current combination
            tool_names = [tool_1['tool']] + [tool['tool'] for tool in combination]

            print(f"当前叠加工具组合为: {tool_names}")

            # Call the test function with the current combination of tools
            test_agent, blue2 ,sample_data= self.test(tool_combination, tool_1['tool'], tool_names[-1])
            
            with open(f'./Result/molecule_captioning_sample_{tool_names}.json','w',encoding='utf-8') as f:
                json.dump(sample_data,f,indent=4)

            # Append the result to the result list
            result_list.append({'agent_tool': test_agent, 'score': blue2, 'tool':tool_names})
            print(f"当前叠加工具组合的分数为: {blue2}")
        result_list = sorted_tools(result_list)
        return result_list
    def _run(self):
        tool_list = self.all_tools
        warmup_result_list = self.warmup
        # print(warmup_result_list)
        top_score = warmup_result_list[0]['score']
        layer = 1
        only_one = False
        result_list = []
        print('\n\033[31m ----工具叠加阶段开始---- \033[0m\n')
        while True:
            print(f'\033[34m --当前工具叠加第{layer}层-- \033[0m')
            if layer == 1:
                last_result_list = warmup_result_list
                if len(last_result_list) == 1:
                    print('由于预热阶段只存在一个工具，结束叠加')
                    only_one = True
                    break
                if len(last_result_list) == 2 and len(tool_list) == 1:
                    print('由于预热阶段只存在一个工具叠加且只叠加了一个工具，则工具叠加与预热阶段第二层将会重复，结束叠加')
                    only_one = True
                    break
                result_list = self.one_Stacking(last_result_list)
            else:
                last_result_list = result_list
                result_list = self.one_Stacking(last_result_list)
                
            layer +=1

            if result_list[0]['score'] <= top_score:
                print(f'第{layer}叠加分数低于前一层最高分数，结束叠加')
                result_list = sorted_tools(last_result_list)
                break
            top_score = result_list[0]['score']
            print(f'第{layer}层最高分数为{top_score},结束叠加进入下一层')

            for i in result_list:
                with open(f'./Result/molecule_captioning_sample_{i["tool"]}.json','r',encoding='utf-8') as f:
                    sample_data = json.load(f)
                i['agent_tool'] = Agent_tool([i['agent_tool']],data=sample_data)

            result_list = last_result_list + result_list
            result_list = sorted_tools(result_list)

        print('\n\033[31m ----工具叠加阶段结束---- \033[0m\n')
        if only_one:
            result_list = sorted_tools(warmup_result_list)
        print("\n\033[34m最终叠加结果为：\033[0m")
        for index,i in enumerate(result_list):
            print(f"{index+1}:{i}")

        return result_list,top_score
    
if __name__ == '__main__':
    # with open('./Dataset/molecule_captioning_test.json','r',encoding='utf-8')    as f:
    #     data = json.load(f)
    # data = data[:1]
    tools = [Name2SMILES(),ChemDFM()]
    top_n = 5
    stack,_ = Stacking(task='',tools=tools,top_n=top_n,tool_number=2)._run()
    print(stack,_)