import inspect
from datetime import datetime
import nltk.translate.bleu_score as bleu
import warnings

def function_to_json(func) -> dict:
    """
    Converts a Python function or method into a JSON-serializable dictionary
    that describes the function's signature, including its name, description, and parameters.
    
    We exclude **tool_args from the parameters.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    # Check if it's a function
    if inspect.isfunction(func):
        signature = inspect.signature(func)
        tool_name = func.__name__
        tool_description = func.__doc__ or ""
    
    # Check if it's a method (bound to a class instance)
    elif isinstance(func, (staticmethod, classmethod)):
        signature = inspect.signature(func.__func__)
        tool_name = func.__func__.__name__
        tool_description = func.__doc__ or ""
    
    # Check if it's a method (instance method)
    elif inspect.ismethod(func):
        signature = inspect.signature(func)
        tool_name = func.__name__
        tool_description = func.__doc__ or ""
    
    # Check if it's a class (constructor method)
    elif inspect.isclass(func):
        # If it's a class, use the signature of the __init__ method
        init_method = func.__init__ if hasattr(func, '__init__') else None
        if init_method:
            signature = inspect.signature(init_method)
        else:
            signature = inspect.Signature()
        tool_name = func.__name__
        tool_description = func.__doc__ or ""
    
    else:
        raise TypeError(f"Provided input is neither a function, method, nor a class: {type(func)}")

    # Process the signature to extract parameters
    parameters = {}
    for param in signature.parameters.values():
        # Skip 'self' for instance methods and exclude **tool_args
        if param.name == 'self' or param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        param_type = type_map.get(param.annotation, "string")  # Default to "string"
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty and param.name != 'self' and param.kind != inspect.Parameter.VAR_KEYWORD
    ]

    # Return the function or method signature in a JSON serializable format
    return {
        "tool_name": tool_name,
        "tool_description": tool_description,
        "parameters": {
            "type": "object",
            "properties": parameters,
            "required": required,
        },
    }


def extract_instance_params(instance):
    """
    Extracts the parameters of an instance object by inspecting its __init__ method
    """
    init_signature = inspect.signature(instance.__init__)
    init_params = {}
    for param_name, param in init_signature.parameters.items():
        if param_name == 'self':
            continue
        param_value = getattr(instance, param_name, None)  
        if param_value not in [None, {}]:
            init_params[param_name] = param_value
            
    return init_params


def test(query:str):
    """
    This is a test function
    """
    return query


def calculate_BLEU(generated_summary, reference_summary, n):
    warnings.filterwarnings("ignore", category=UserWarning)

    # Tokenize the generated summary and reference summary
    generated_tokens = list(generated_summary)
    reference_tokens = list(reference_summary)

    # Calculate the BLEU score
    weights = [1.0 / n] * n  # Weights for n-gram precision calculation
    bleu_score = bleu.sentence_bleu([reference_tokens], generated_tokens, weights=weights)
    return bleu_score

def sorted_tools(list):
    return sorted(list, key=lambda x: x['score'], reverse=True)


def task2query(task:str):
    if task == "Query2SMILES":
        query="Question: Please try to infer the SMILES of this molecule:\nDescription:"
        description = "Input a molecule's description ,returns a SMILES. Note:1.the results returned by this tool may not necessarily be correct. 2.The input description try to be the same with the original description."
    elif task == "SMILES2Query":
        query="Question: Please show me the detailed description of more than 20 words of the given SMILES:\nSMILES:"
        description = "Input a molecule's SMILES ,returns a description. Note:1.the results returned by this tool may not necessarily be correct. 2.The input description try to be the same with the original SMILES."
    elif 'MolecularPropertyPrediction' in task:
        task_name = task.split('_')[1]
        description="Input a molecule's SMILES ,returns the answer. Note:1.the results returned by this tool may not necessarily be correct. 2.The input SMILES try to be the same with the original SMILES."
        if task_name == 'bace':
            query="Given a molecular structure represented by a SMILES string, please analyze whether this compound can act as a BACE1 (Beta-site Amyloid Precursor Protein Cleaving Enzyme 1) inhibitor. Considering the molecule's structural features including molecular weight, atom composition, bond types, and functional groups to determine its potential BACE1 inhibitory activity. This prediction is crucial for identifying potential therapeutic agents for Alzheimer's disease treatment. Please provide your prediction as a simple Yes (can inhibit BACE1) or No (cannot inhibit BACE1).\nSMILES:"
        elif task_name == 'bbbp':
            query="Given a molecular structure represented by a SMILES string, please analyze whether this compound can penetrate the blood-brain barrier (BBB). Considering the molecule's structural features including molecular weight, atom composition, bond types, functional groups, lipophilicity, and other physicochemical properties that influence BBB penetration. This prediction is crucial for developing drugs that need to reach the central nervous system. Please provide your prediction as a simple Yes (can penetrate BBB) or No (cannot penetrate BBB).\nSMILES:"
        elif task_name == 'clintox':
            query="Given a molecular structure represented by a SMILES string, please analyze whether this compound has received FDA approval or failed in clinical trials specifically due to toxicity issues. Considering the molecule's structural features including molecular weight, atom composition, bond types, functional groups, and potential toxicophores that might influence its safety profile. This prediction is crucial for early assessment of drug candidates' potential success in clinical trials. Please provide your prediction as a simple Yes (FDA approved) or No (failed due to toxicity).\nSMILES:"
        elif task_name == 'hiv':
            query="Given a molecular structure represented by a SMILES string, please analyze whether this compound has the ability to inhibit HIV replication. Considering the molecule's structural features including molecular weight, atom composition, bond types, functional groups, and other physicochemical properties that might influence its anti-HIV activity. This prediction is crucial for identifying potential therapeutic agents for HIV treatment. Please provide your prediction as a simple Yes (can inhibit HIV replication) or No (cannot inhibit HIV replication).\nSMILES:"
        elif task_name == 'tox21':
            query="Given a molecular structure represented by a SMILES string, please analyze whether this compound acts as an agonist or antagonist of the nuclear receptor androgen receptor (NR-AR). Considering the molecule's structural features including molecular weight, atom composition, bond types, functional groups, and structural characteristics that determine its interaction with NR-AR. This prediction is crucial for understanding the compound's potential endocrine-related activities. Please provide your prediction as a simple Yes (agonist) or No (antagonist)."
    elif task == "ReactionPrediction":
        query = "Given an incomplete chemical reaction equation in SMILES notation (format: reactants>>products, where multiple reactants are separated by dots '.'), predict and complete the missing products marked as '___'. The response should only contain the SMILES representation of the missing molecule, without any additional explanation. Several examples will be provided \nExample1:{'reaction':'BrBr.c1ccc2cc3ccccc3cc2c1>>___','answer':'Brc1c2ccccc2cc2ccccc12'}.\nExample2:{'reaction':'CN.O=C(O)c1ccc(Cl)c([N+](=O)[O-])c1>>___','answer':'CNc1ccc(C(=O)O)cc1[N+](=O)[O-]'}\nChemical reaction equation:"
        description = "Input an incomplete chemical reaction ,returns the answer. Note:1.the results returned by this tool may not necessarily be correct. 2.The input reaction try to be the same with the original reaction."
    elif task == "YieldPrediction":
        query= "Given the SMILES string representation of a Buchwald-Hartwig reaction (format: reactants>>products, where multiple reactants are separated by dots '.'), can you predict if the reaction is High-yielding (Yes) or Not High-yielding (No) based on whether the yield rate is above 70%? Answer with only Yes or No. Example will be provided.\nExample:{'reaction':'FC(F)(F)c1ccc(Br)cc1.Cc1ccc(N)cc1.O=S(=O)(O[Pd]1c2ccccc2-c2ccccc2N-1)C(F)(F)F.CC(C)c1cc(C(C)C)c(-c2ccccc2P(C(C)(C)C)C(C)(C)C)c(C(C)C)c1.CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C.CCOC(=O)c1cnoc1C>>Cc1ccc(Nc2ccc(C(F)(F)F)cc2)cc1','answer':'No'} \nReaction:"
        description = "Input a Buchwald-Hartwig reaction ,returns the answer. Note:1.the results returned by this tool may not necessarily be correct. 2.The input reaction try to be the same with the original reaction."
    return query, description


if __name__=='__main__':
    list = [{'score':0.5},{'score':0.6},{'score':0.7}]
    print(sorted_tools(list))