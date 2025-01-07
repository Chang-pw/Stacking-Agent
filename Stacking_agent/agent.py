from .Basemodel import ChatModel
from .Tool import Tools
from .ReAct_prompt import REACT_PROMPT, TOOL_DESC
from .utils import *
from typing import Dict, List, Optional, Tuple, Union
import json5
import re
import time
class Agent:
    name :str = "Agent"
    def __init__(self, all_tools:dict = []):
        self.tool = Tools(all_tools)
        self.toolcon = self.tool.toolConfig
        self.model = ChatModel()
        self.system_prompt = self.build_system_input()
        self.initcon = self.tool.initConfig
    def build_system_input(self):
        """construct system prompt for agent"""
        tool_descs, tool_names = [], []
        for tool in self.toolcon:
            tool_descs.append(TOOL_DESC.format(**tool))
            tool_names.append(tool["tool_name"])
        tool_descs = "\n\n".join(tool_descs)
        tool_names = ",".join(tool_names)
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt

    def parse_latest_plugin_call(self, text):
        """parse the latest tool call and parameters"""
        text = str(text)
        plugin_name, plugin_args = '', ''
        i = text.rfind("\033[93mAction:\033[0m")
        j = text.rfind("\033[94mAction Input:\033[0m")
        k = text.rfind("\033[96mObservation:\033[0m")
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                text = text.rstrip() + "\033[96mObservation:\033[0m"  # Add it back.
            k = text.rfind("\033[96mObservation:\033[0m")
            plugin_name = text[i + len("\033[93mAction:\033[0m") : j].strip()
            plugin_args = text[j + len("\033[94mAction Input:\033[0m") : k].strip()
            text = text[:k]
        return plugin_name, plugin_args, text
        
    def call_plugin(self, plugin_name, plugin_args):
        '''call the tool'''
        try:
            plugin_args = json5.loads(plugin_args)
        except:
            plugin_args = {'query': plugin_args}
            # return f"\n\033[96mObservation: \033[0m Tool call failed with error: Invalid plugin arguments format"
        try:
            # Check if the plugin_args is valid
            if not isinstance(plugin_args, dict):
                raise ValueError(f"Invalid plugin arguments format: {plugin_args}")
            
            # Add necessary configurations
            for i in self.initcon:
                if i["tool_name"] == plugin_name:
                    plugin_args.update({k: v for k, v in i.items() if k != "tool_name"})
        except Exception as e:
            # Catch the exception and return error message
            return f"\n\033[96mObservation: \033[0m Tool call failed with error: {str(e)}"

        try:
            return "\n\033[96mObservation: \033[0m" + self.tool(plugin_name, **plugin_args)
        except Exception as e:
            # Catch the exception and return error message
            return f"\n\033[96mObservation: \033[0m Tool call failed with error: {str(e)}"

    def text_completion(self, text, history=[]):
        text = "\nQuestion:" + text
        while True:
            response = ""
            response, his = self.model.chat(text, history, self.system_prompt,stop_word="Observation")
            if response == "Run Again":
                continue
            break
        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)
        if plugin_name:
            # print(plugin_name,plugin_args)
            response += self.call_plugin(plugin_name, plugin_args)
            # print('\033[91m' + 'Tool call:' + '\033[0m', plugin_name, '\033[91m' + 'Tool Para:'+ '\033[0m',plugin_args)        # # print(response)
        # response, his = self.model.chat(response, history, self.system_prompt)
        return response, his
    
    def _run(self, text, history=[],debug=True):
        
        if debug == True:
            print('\033[91m ============================START============================ \033[0m')
        max_iter = 10
        n=0
        while n<=max_iter:
            time.sleep(2)
            n+=1
            response, his = self.text_completion(text, history)
            if debug == True:
                print(response)
            if 'Final Answer:' in response:
                parts = response.split("Final Answer:")
                final_answer = parts[1].strip()
                if '\u001b[0m' in final_answer:
                    final_answer =final_answer.split('\u001b[0m')[-1].strip()
                if debug == True:
                    print("\033[91m =============================END=============================\033[0m")
                return final_answer, response, history
            text += '\n' + response
            
        return 'Error','Error','Error'

    
if __name__ == '__main__':
    agent = Agent()
    query = 'The molecule is a member of the class of cyclopentanols carrying 1,2,4-triazol-1-ylmethyl and 4-chlorobenzylidene and geminal dimethyl substituents at positions 1, 2 and 5 respectively. It is a member of triazoles, a member of monochlorobenzenes, a member of cyclopentanols, a tertiary alcohol and an olefinic compound.'
    query = query + 'Please try to infer the SMILES of this molecule.'
    # print(agent.system_prompt)
    answer,response, _ = agent._run(text=query, history=[])
    print(answer)