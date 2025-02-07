Task='Molecule_Design'
topN=5
tool_number=2
tools='''
[
Name2SMILES(),
ChemDFM()
]
'''
# python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"
# tool_number=3
# python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"
# tool_number=4
# python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"

# Task='Molecule_captioning'
# topN=5
# tool_number=2
# tools='''
# [
# SMILES2Description(),
# ChemDFM()
# ]
# '''
# python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"
# tool_number=3
# python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"
# tool_number=4
# python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"

Task='MolecularPropertyPrediction_bace'
topN=5
tool_number=2
tools='''
[
SMILES2Property(),
ChemDFM()
]
'''
python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"
tool_number=3
python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"
tool_number=4
python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"
