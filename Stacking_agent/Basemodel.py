import os
from openai import AzureOpenAI
from typing import List, Dict
import time
class ChatModel():
    def __init__(self, model="gpt-4o", temperature=0.7):
        self.model = model
        self.temperature = temperature
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2024-08-01-preview"
        os.environ["OPENAI_API_BASE"] = "https://xiaojin.openai.azure.com/openai"
        os.environ["OPENAI_API_ENDPOINT"] = "https://xiaojin.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-08-01-preview"
        os.environ["OPENAI_API_KEY"] = "8f09ac0303fd4a8584796c08a5013fea"
        os.environ["OPENAI_GPT4O_DEPLOYMENT_NAME"] = "gpt4o"
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
        )
    def chat(self, prompt: str, history: List[Dict[str, str]], system_prompt: str = 'You are a helpful assistant',stop_word:str='') -> str:
        """
        Get response with the prompt,history and system prompt.

        Args:
            prompt (str)
            history (List[Dict[str, str]])
            system_prompt (str)

        """


        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        

        # for entry in history:
        #     messages.append(entry)

        messages.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stop = stop_word
            )
        except Exception as e:
            print(e)
            return "Run Again",history

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
        
        return response,history
    def test(self):
        print('test')

if __name__ == '__main__':
    gpt4o = ChatModel()
    print(gpt4o.chat('Please tell me the food of china', []))
    

