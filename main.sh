## Task: 'Molecule Generation'='Query2SMILES';'Molecule Caption'='SMILES2Query'
## Tools: 'Name2SMILES','ChemDFM'
## TopN: Selection of the best performing N tools for stacking
## Tool Number: Number of tools each agent can call
## Train data number: Number of training data

Task='Query2SMILES'
tools='''
[
Name2SMILES(),
# ChemDFM()
]
'''
topN=5
tool_number=2
train_data_number=10

python main.py --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number" 