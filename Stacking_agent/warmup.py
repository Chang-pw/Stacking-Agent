from .agent import Agent
from .utils import *
from .tools import *
import random
import json
import time
from tqdm import tqdm
import concurrent.futures


class Warmup:
    def __init__(self,tools=[],tool_number=2,data=[],train_data_number=10,query=""):
        self.seed = 2025
        self.tool = tools
        self.tool_number = tool_number
        self.data = data
        self.train_data_number = train_data_number
        self.task_query = query
        self.sample_data = self.sample()
    def sample(self):
        # random.seed(self.seed)
        sample_data = random.sample(self.data,self.train_data_number)
        return sample_data

    def test(self,tool=[],wo_agent=False):
        agent = Agent(tool)
        sample_data = self.sample_data
        score = 0
        for i in sample_data:
            smiles = i["SMILES"]
            description = i["description"]
            query = self.task_query + description
            if wo_agent:
                final_answer = tool[0].wo_run(query)
                agent = tool[0]
            else:
                final_answer, response, history = agent._run(query,[],debug=True)
            i["answer"] = final_answer
            i["blue2"] = calculate_BLEU(final_answer,smiles,2)
            score += i["blue2"]
            time.sleep(5)
        blue2 = score/len(sample_data)

        return agent,blue2,sample_data
    
    def one_tool_stacking(self,tool:dict):
        name = str(tool[0])
        layer = 0
        score = -1
        Tool_agent= []
        tool_list = tool
        Score_list = []
        while True:
            if layer == 0:
                wo_agent = True
                test_agent,blue2,sample_data = self.test(tool_list,wo_agent=wo_agent)
            elif layer == 1:
                wo_agent = False
                test_agent,blue2,sample_data = self.test(tool_list)
            else:
                Tool_agent_list = Tool_agent[1:][-(self.tool_number-1):]
                test_agent,blue2,sample_data = self.test(tool_list+Tool_agent_list)
            print(f"{name}叠加的第{layer}层的blue2为{blue2}")
            if blue2 > score:
                with open(f"./Result/molecule_captioning_sample_{name}_{layer}.json","w",encoding="utf-8") as f:
                    json.dump(sample_data,f,indent=4)
                score = blue2
                if wo_agent:
                    Agent_t = tool[0]
                else:
                    Agent_t = Agent_tool(test_agent,data=sample_data)
                Tool_agent.append(Agent_t)
                Score_list.append(blue2)
                layer +=1
                if score == 1:
                    break
            else:
                break
        return Tool_agent,Score_list

    def _run(self):
        tool_list = self.tool
        result_list = []
        print("\033[31m ----工具预热阶段开始---- \033[0m\n")
        # 使用ThreadPoolExecutor并行化工具叠加任务
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 为每个工具在tool_list中创建一个任务
            futures = [executor.submit(self.one_tool_stacking, [tool]) for tool in tool_list]
            
            # 等待所有任务完成并处理结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    # 获取每个任务的返回值（tool_agent, score_list）
                    tool_agent, score_list = future.result()
                    
                    tool_name = tool_list[futures.index(future)]
                    print(f"{tool_name}的叠加完成")
                    print(f"{tool_name}的最高blue2为{score_list[-1]}")
                    
                    layer = 0
                    for agent, score in zip(tool_agent, score_list):
                        result_list.append({"agent_tool": agent, "score": score, "tool": f"{tool_name}_{layer}"})
                        layer += 1

                except Exception as e:
                    print(f"处理工具时发生错误: {e}")


        print("\n\033[31m ----工具预热阶段结束---- \033[0m")
        print("\n\033[34m可使用工具为：\033[0m")
        for index,i in enumerate(result_list):
            print(f"{index+1}:{i}")
        return result_list
    
if __name__ == "__main__":
    # tool_list = [Name2SMILES()]
    # warmup = Warmup(tool_list)
    # print(warmup.one_tool_stacking(tool_list))
    with open("./Dataset/Description2SMILES_train_data.json","r",encoding="utf-8") as f:
        train_data = json.load(f)
    all_tools = [Name2SMILES()]

    warmup = Warmup(all_tools,data=train_data,tool_number=2,train_data_number=10,query=task2query('Query2SMILES'))._run()
    # print(warmup)
     