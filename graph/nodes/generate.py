

from typing import Any, Dict
from graph.state import GraphState
from graph.chains.generation import generation_chain


def generate(state: GraphState) -> Dict[str, Any]:
    question = state['question']
    documents = state['documents']

    response = generation_chain.invoke({"question": question, "context": documents})
    return {"question": question, "documents": documents, "generation": response}
    
