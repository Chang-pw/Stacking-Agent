import openai
import json
from tqdm import tqdm


# my_dict1 = {i['SMILES']:i['description'] for i in test_data}


def get_response(query):
    client = openai.OpenAI(
        api_key='kw-NJ9rRVi6zd3eLZb92Hdp9nRgriwb2UmKWqEdqfA7Uaj5qDXE',
        base_url='http://10.88.3.81:8502'
    )
    request_params = {
        "model": "gpt-4o", # gpt-4o, gpt-4, claude-3-5-sonnet-20241022, ... （见本页面上方“可用模型说明”）
        # 下面可以多搞点东西，确保能够正常运行
        "messages": [
            {"role": "system", "content": "You are an expert chemist."},
            {"role": "user", "content": query},
        ],
    }
    response = client.chat.completions.create(**request_params)
    response_dict = response.model_dump()
    return response_dict['choices'][0]['message']['content']

# prompt = """You are an expert chemist. Given the molecular SMILES, your task is to provide the detailed description of the molecule using your experienced chemical Molecular knowledge. 
# Please strictly follow the format, no other information can be provided.

# Molecular SMILES:{query}
# Molecular Description:
# """

# prompt = """You are an expert chemist. Given the molecular SMILES, your task is to provide the detailed description of the molecule using your experienced chemical Molecular knowledge. 
# Please strictly follow the format, no other information can be provided.

# """
# format = """Molecular SMILES:{SMILES}
# Molecular Description:{description}
# """


import random
data = []
with open('/data1/bowei/agent/my_agent/Stacking_agent/prompt/ReactionPrediction_reagent_GPT4o_prompt.json','r',encoding='utf-8') as f:
    test_data = json.load(f)
for i in tqdm(test_data):
    # smiles = i['SMILES']
    # gold_answer = i['gold_answer']
    # reaction = i['reaction']
    # test_prompt=prompt.format(Reaction=reaction)
    answer = get_response(i).split('Optimal reactant:')[-1].strip()
    print(answer)
    data.append({'prompt':i,'answer':answer})
    with open('./Result/Stacking/ReagentSelection_reactant/GPT4o_0shot.json','w',encoding='utf-8') as f:
        json.dump(data,f,indent=2)
# for i in tqdm(test_data[90:]):
#     smiles = i['SMILES'] 
#     test_prompt = prompt
#     description = i['description']
#     random_keys = random.sample(list(my_dict.keys()), 10)
#     for j in random_keys:
#         test_prompt+=format.format(description=my_dict[j],SMILES=j)
#     # test_prompt=prompt.format(query=smiles)
#     test_prompt+=format.format(description='',SMILES=smiles)
#     new_prompt.append(test_prompt)
#     # print(test_prompt)
#     answer = get_response(test_prompt)
#     i['answer'] = answer
#     print(answer)

#     with open('./Result/SMILES2Query/GPT4o_random_10shot.json','w',encoding='utf-8') as f:
#         json.dump(test_data,f,indent=2)
#     with open('./Stacking_agent/prompt/SMILES2Query_GPT4o_random_10shot_prompt.json','w',encoding='utf-8') as f:
#         json.dump(new_prompt,f,indent=2)
# with open('/data1/bowei/agent/my_agent/Stacking_agent/prompt/SMILES2Query_GPT4o_scaffold_10shot_prompt.json','r',encoding='utf-8') as f:
#     new_prompt = json.load(f)
# new_data=[]

# for i in tqdm(new_prompt):
#     last_index = i.rfind('\nMolecule SMILES: ')
#     if last_index != -1:
#         # 从最后一个出现的位置提取子字符串
#         result = i[last_index + len('\nMolecule SMILES: '):].strip()  # 从找到的位置开始，去掉前导空格
#         # print(result)
#     answer = get_response(i)
    
#     print(answer)
#     new_data.append({'SMILES':result.split('\n')[0],'description':my_dict[result.split('\n')[0]],'answer':answer})
#     print(my_dict[result.split('\n')[0]])
#     with open('./Result/SMILES2Query/GPT4o_scaffold_10shot.json','w',encoding='utf-8') as f:
#         json.dump(new_data,f,indent=2)