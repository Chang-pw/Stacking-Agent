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
        query="Question: Please show me a description of this molecule::\nSMILES:"
        description = "Description:"
    

    return query, description


if __name__=='__main__':
    list = [{'score':0.5},{'score':0.6},{'score':0.7}]
    print(sorted_tools(list))