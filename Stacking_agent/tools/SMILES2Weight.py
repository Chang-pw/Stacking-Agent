from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
import requests
from ..Basemodel import ChatModel


class SMILES2Weight():
    name: str = "SMILES2Weight"
    description: str = "Input SMILES, returns molecular weight."

    def __init__(
        self,
    ):
        super(SMILES2Weight, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        return mol_weight

    def __str__(self):
        return "SMILES2Weight tool"

    def __repr__(self):
        return self.__str__()

    async def _arun(self, smiles: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
    
    def wo_run(self,query,debug=False):
        model = ChatModel()
        prompt = "Please output only one SMILES name for use in the weight of molecule based on the question:" + query
        response,history = model.chat(prompt=prompt,history=[])
        answer = self._run(response)
        if answer == "Could not find a molecule matching the text. One possible cause is that the input is incorrect, please modify your input.":
            return ""
        return answer
    
