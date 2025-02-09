import os
from openai import AzureOpenAI
from typing import List, Dict
import time
import random
class ChatModel():
    def __init__(self, model="gpt-4o", temperature=0.7):
        self.model = model
        self.temperature = temperature

    def chat(self, prompt: str, history: List[Dict[str, str]], system_prompt: str = 'You are a helpful assistant',stop_word:str='') -> str:
        """
        Get response with the prompt,history and system prompt.

        Args:
            prompt (str)
            history (List[Dict[str, str]])
            system_prompt (str)

        """
        chosen_index = random.randint(0, len(self.api_keys) - 1)
        chosen_key = self.api_keys[chosen_index]
        chosen_endpoint = self.api_endpoints[chosen_index]
        
        client = AzureOpenAI(
            api_key=chosen_key,
            api_version="2024-08-01-preview",
            azure_endpoint=chosen_endpoint
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        

        # for entry in history:
        #     messages.append(entry)

        messages.append({"role": "user", "content": prompt})
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stop = stop_word
            )
        except Exception as e:
            print(e)
            return "Run Again",history
        total_tokens = response.usage.total_tokens
        response = response.choices[0].message.content
        history.append({"role": "assistant", "content": response})
        try:
            response = response.replace("Thought:","\033[92mThought:\033[0m")
            response = response.replace("Action:","\033[93mAction:\033[0m")
            response = response.replace("Action Input:","\033[94mAction Input:\033[0m")
        except:
            pass
        try:
            response = response.replace("Final Answer:","\033[91mFinal Answer:\033[0m")
        except:
            pass
        
        return response,total_tokens
    def test(self):
        print('test')

if __name__ == '__main__':
    gpt4o = ChatModel()
    a,b = gpt4o.chat('Hello', [])
    
    
    

