from MultiAgent.Structures.Chain import ChainChat
from Stacking_agent.agent import Agent 
from Stacking_agent.tools import *
from Stacking_agent.utils import *
from Stacking_agent.generator import *


class MultiAgent:
    def __init__(self):
        pass

    @staticmethod
    def run(agent, prompt='', model='', num=1):
        if model == "Chain":
            model = ChainChat()  # 创建一个新的ChainChat实例
        
        model.add_agent(num,agent)  # 使用传入的model
        final_answer, answers = model.run(prompt)
        return final_answer,answers