import threading

class Agent_tool:
    instance_count = 0
    instance_lock = threading.Lock()  
    description :str = '''
    Input a question, returns SMILES. 
    Note: 1.the results returned by this tool may not necessarily be correct.
    2.The question should be the same with the Descritpion
    '''
    def __init__(self,agent,data=[],buffer=True,next_n=True,debug=False,**tool_args):
        with Agent_tool.instance_lock:  # 加锁，确保线程安全
            if next_n:
                Agent_tool.instance_count += 1
            self.number = Agent_tool.instance_count
        self.agent = agent
        self.buffer = buffer
        self.data = data
        self.name = f'Query2SMILES_{self.number}'
        self.debug = debug

    def _run(self,query:str,**tool_args)->str:
        if self.buffer == True:
            final = self.buffer_run(query,**tool_args)
            if final != None:
                return final
        final,response,hs = self.agent._run(query,[],debug=self.debug)
        return final
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()
    
    def buffer_run(self,query:str,**tool_args)->str:
        query_dict = {i['description']:i['answer'] for i in self.data}
        if query in query_dict.keys():
            # print('使用缓存')
            return query_dict[query]
        else:
            return "Error:please modify your input"

if __name__ == '__main__':
    print(str(Agent_tool(1)))