import requests
from rdkit import Chem
from ..Basemodel import ChatModel


def is_smiles(text):
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False
        return True
    except:
        return False

def largest_mol(smiles):
    ss = smiles.split(".")
    ss.sort(key=lambda a: len(a))
    while not is_smiles(ss[-1]):
        rm = ss[-1]
        ss.remove(rm)
    return ss[-1]

def extract_property(data):
    compound = data['PC_Compounds'][0]
    properties = {}
    for i in compound:
        if i == 'props':
            for prop in compound['props']:
                label = prop['urn']['label']
                name = prop['urn'].get('name', '')

                # 获取值
                value = next((v for k, v in prop['value'].items()), None)

                # 创建属性键名
                key = f"{label}_{name}" if name else label
                properties[key] = value
            continue
        properties[i] = compound[i]

    return properties


class Name2Property:
    name: str = "Name2Property"
    description: str = "Input only one compounds name, returns Property. "
    def __init__(self, **tool_args):
        pass
    
    def _run(self, query: str,**tool_args) -> str:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
        # Query the PubChem database
        r = requests.get(url.format(query, "/JSON"))
        # Convert the response to a JSON object
        data = r.json()
        try:
            smi = data
        except:
            return "Could not find a molecule matching the text. One possible cause is that the input is incorrect, please modify your input."
        try:
            smi = extract_property(smi)
        except:
            pass
        return str(smi)
    
    def __str__(self):
        return "Name2Property tool"

    def __repr__(self):
        return self.__str__()

    def wo_run(self,query,debug=False):
        # model = ChatModel()
        # prompt = "Please output only one molecule name for based on the question:" + query
        # response,history = model.chat(prompt=prompt,history=[])
        # answer = self._run(response)
        # if answer == "Could not find a molecule matching the text. One possible cause is that the input is incorrect, please modify your input and the input needs to be a moleculer name not a SMILES.":
        #     return ""
        # return answer
        return ""  ## It can not solve the question without agent
